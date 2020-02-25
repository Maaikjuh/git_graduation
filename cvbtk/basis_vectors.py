# -*- coding: utf-8 -*-
"""
This module provides classes that define alternative basis vectors.
"""
import os

import numpy as np
from dolfin import DirichletBC, Constant, TrialFunction, TestFunction, dot, grad, dx, Function, solve, assemble, inner, \
    LocalSolver, Cell, Expression, as_matrix, cos, sin, Mesh, VectorFunctionSpace, sqrt, tan, FunctionSpace, \
    interpolate, SpatialCoordinate, as_vector, unit_vector, cross, project
from dolfin.cpp._common import mpi_comm_self
from dolfin.cpp.common import MPI, mpi_comm_world, Parameters
from dolfin.cpp.io import HDF5File
from dolfin.cpp.mesh import Point

from .coordinate_systems import CoordinateSystem
from .utils import vector_space_to_scalar_space, print_once, atan2_, reset_values, save_to_disk
from numpy.linalg import norm
import scipy.integrate
import scipy

__all__ = [
    'CardiacBasisVectors',
    'BasisVectorsBayer',
    'CylindricalBasisVectors',
    'CylindricalBasisVectorsBiV',
    'FiberBasisVectors',
    'GeoFunc'
]


class CardiacBasisVectors(object):
    """
    Cardiac basis vectors defined by circumferential (ec), longitudinal (el),
    and transmural (et) values.

    https://www.ncbi.nlm.nih.gov/pubmed/19592607
    """

    @staticmethod
    def ec(V, sig, tau, phi, focus):
        """
        Cardiac basis vector oriented circumferentially around the z-axis.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            sig: Ellipsoidal radial position σ.
            tau: Ellipsoidal longitudinal position τ.
            phi: Ellipsoidal circumferential position φ.
            focus: Distance from the origin to the shared focus points.

        Returns:
            :class:`~numpy.array`
        """
        # TODO Check that V is in proper space.
        Q = vector_space_to_scalar_space(V)

        # Extract the coordinate values of sigma, tau, and phi in array form.
        sig_array = sig.to_array(Q)
        tau_array = tau.to_array(Q)
        phi_array = phi.to_array(Q)

        # Compute shared component terms.
        term = np.sqrt((sig_array**2 - 1)*(1 - tau_array**2))

        # Compute the x and y components of ec. The z component is zeroes.
        ec_x = -focus*term*np.sin(phi_array)
        ec_y = focus*term*np.cos(phi_array)
        ec_z = np.zeros(len(ec_x))

        # Compute the normalization factor as an array.
        norm = np.sqrt(ec_x**2 + ec_y**2)

        # Normalize the relevant components and return.
        ec_x /= norm
        ec_y /= norm
        return ec_x, ec_y, ec_z

    @staticmethod
    def el(V, sig, tau, phi, focus):
        """
        Cardiac basis vector oriented longitudinally along the ellipsoidal
        surface(s).

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            sig: Ellipsoidal radial position σ.
            tau: Ellipsoidal longitudinal position τ.
            phi: Ellipsoidal circumferential position φ.
            focus: Distance from the origin to the shared focus points.

        Returns:
            :class:`~numpy.array`
        """
        # TODO Check that V is in proper space.
        Q = vector_space_to_scalar_space(V)

        # Extract the coordinate values of sigma, tau, and phi in array form.
        sig_array = sig.to_array(Q)
        tau_array = tau.to_array(Q)
        phi_array = phi.to_array(Q)

        # Compute shared component terms.
        term = np.sqrt((sig_array**2 - 1)*(1 - tau_array**2))
        beta = (tau_array - sig_array**2*tau_array)/term

        # Compute the x, y, and z components of el.
        el_x = focus*beta*np.cos(phi_array)
        el_y = focus*beta*np.sin(phi_array)
        el_z = focus*sig_array

        # Compute the normalization factor as an array.
        norm = np.sqrt(el_x**2 + el_y**2 + el_z**2)

        # Normalize the relevant components and return.
        el_x /= norm
        el_y /= norm
        el_z /= norm
        return el_x, el_y, el_z

    @staticmethod
    def et(V, sig, tau, phi, focus):
        """
        Cardiac basis vector oriented transmurally normal to the inner and outer
        surfaces.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            sig: Ellipsoidal radial position σ.
            tau: Ellipsoidal longitudinal position τ.
            phi: Ellipsoidal circumferential position φ.
            focus: Distance from the origin to the shared focus points.

        Returns:
            :class:`~numpy.array`
        """
        # TODO Check that V is in proper space.
        Q = vector_space_to_scalar_space(V)

        # Extract the coordinate values of sigma, tau, and phi in array form.
        sig_array = sig.to_array(Q)
        tau_array = tau.to_array(Q)
        phi_array = phi.to_array(Q)

        # Compute shared component terms.
        term = np.sqrt((sig_array**2 - 1)*(1 - tau_array**2))
        beta = (sig_array - sig_array*tau_array**2)/term

        # Compute the x, y, and z components of et.
        et_x = focus*beta*np.cos(phi_array)
        et_y = focus*beta*np.sin(phi_array)
        et_z = focus*tau_array

        # Compute the normalization factor as an array.
        norm = np.sqrt(et_x**2 + et_y**2 + et_z**2)

        # Normalize the relevant components and return.
        et_x /= norm
        et_y /= norm
        et_z /= norm
        return et_x, et_y, et_z

class QuatMath(object):
    """
    Contains functions for quaternion mathematics.
    """

    @staticmethod
    def quatmul(quaternion1, quaternion0):
        """
        Multiplication of 2 quaternions.
        As implemented by https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
        """
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    @staticmethod
    def quat2rot(q):
        """
        Converts a unit quaternion to the corresponding rotation matrix.
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        Args:
            q: a numpy array representing a unit(!) quaternion, where q[0] represents
            the scalar and q[1:] the vector part of the quaternion.
        Returns:
            A 3x3 orthogonal matrix
        """
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        rot = np.zeros((3, 3))
        rot[0, 0] = 1 - 2 * (q2 ** 2 + q3 ** 2)
        rot[1, 1] = 1 - 2 * (q1 ** 2 + q3 ** 2)
        rot[2, 2] = 1 - 2 * (q1 ** 2 + q2 ** 2)
        rot[0, 1] = 2 * (q1 * q2 - q0 * q3)
        rot[0, 2] = 2 * (q1 * q3 + q0 * q2)
        rot[1, 0] = 2 * (q1 * q2 + q0 * q3)
        rot[1, 2] = 2 * (q2 * q3 - q0 * q1)
        rot[2, 0] = 2 * (q1 * q3 - q0 * q2)
        rot[2, 1] = 2 * (q2 * q3 + q0 * q1)

        return rot

    @staticmethod
    def rot2quat(rot):
        """
        Converts an orthogonal rotation matrix to a unit quaternion.
        https://www.springer.com/cda/content/document/cda_downloaddocument/9789400761001-c2.pdf?SGWID=0-0-45-1381731-p174773711
        Args:
            rot: a 3x3 numpy array representing the rotation matrix.
        Returns:
            A 4x1 numpy array representing the unit quaternion.
        """

        r11 = rot[0, 0]
        r22 = rot[1, 1]
        r33 = rot[2, 2]
        r12 = rot[0, 1]
        r13 = rot[0, 2]
        r21 = rot[1, 0]
        r23 = rot[1, 2]
        r31 = rot[2, 0]
        r32 = rot[2, 1]

        q = np.zeros(4)
        q[0] = np.sqrt(1 + r11 + r22 + r33)
        q[1] = np.sign(r32 - r23) * np.sqrt(1 + r11 - r22 - r33)
        q[2] = np.sign(r13 - r31) * np.sqrt(1 - r11 + r22 - r33)
        q[3] = np.sign(r21 - r12) * np.sqrt(1 - r11 - r22 + r33)

        return q * 0.5

    @staticmethod
    def slerp(q1, q2, t):
        """
        Spherical linear interpolation between two quaternions q1 and q2 with interpolation parameter t.
        If t=0, returns q1. If t=1, returns q2.
        https://en.wikipedia.org/wiki/Slerp

        Args:
            q1: array representing first quaternion.
            q2: array representing second quaternion.
            t: interpolation parameter.
        Returns:
            array representing the interpolated quaternion.
        """
        # Normalize quaternions to avoid problems when they are not unit quaternions.
        q1 = q1 / norm(q1)
        q2 = q2 / norm(q2)

        # Compute dot product of the two quaternions.
        dot = np.dot(q1, q2)

        # If the dot product is negative, the quaternions have
        # opposite handedness and slerp won't take the shortest
        # path. Fix by reversing one quaternion.
        if dot < 0.0:
            q2 = -1 * q2
            dot = -1 * dot

        if dot > 0.9995:
            # If the inputs are very close, linearly interpolate.
            qi_ = q1 + t * (q2 - q1)
            return qi_ / norm(qi_)

        # Stay within the domain of arccos.
        dot = np.clip(dot, -1, 1)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        s1 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)  # == sin(theta_0 - theta)/sin(theta_0)
        s2 = np.sin(theta) / np.sin(theta_0)

        return (s1 * q1) + (s2 * q2)


class BasisVectorsBayer(object):
    """
    Create cardiac basis vectors defined by circumferential (ec), longitudinal (el), and transmural (et) directions
    and cardiac fiber vectors defined by fiber direction (ef), sheet direction (es), and sheet-normal direction (en)
    as proposed by the method of Bayer et al.

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3518842/

    Args:
        geometry: cvbtk geometry object with attributes mesh(), and tags() [mesh, facet function with boundary markers]
        func_ah_w: Python function for helix angle for free walls. Should take 2 arguments: the longitudinal normalized coordinate (u) and the transmural normalized coordinate (v).
        func_ah_s: Python function for helix angle for septum. Should take 2 arguments: the longitudinal normalized coordinate (u) and the transmural normalized coordinate (v).
                   If not given, the same function as for the wall (func_ah_w) will be used.
        func_at: Python function for transverse angle. Should take 2 arguments: the longitudinal normalized coordinate (u) and the transmural normalized coordinate (v).
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """

    def __init__(self, geometry, func_ah_w=None, func_ah_s=None, func_at=None, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        self._geometry = geometry

        self._func_ah_w = func_ah_w
        if func_ah_s is None:
            func_ah_s = func_ah_w
        self._func_ah_s = func_ah_s
        self._func_at = func_at

        self._geometry_type = self.check_geometry_type()
        self._ec = None
        self._el = None
        self._et = None
        self._ef = None
        self._es = None
        self._en = None
        self._u = None
        self._v_wall = None
        self._v_septum = None
        self._t_endo = None
        self._t_interp = None

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this class.
        By specifying the parameters, it is possible to change between the original and adapted Bayer method.
        Use default parameters for the adapted method.

        Parameters:
            apex_bc_mode: Either 'line' or 'point'.
                          If 'line', the BC for the apex are set to all the vertices of each
                          element that intersects the z-axis (recommended).
                          If 'point', the apex BC is assigned to the lowest point in the mesh
                          (as originally proposed by Bayer et al. 2012).
            correct_ec (boolean): When True, enforce the circumferential cardiac unit vector to be horizontal
                                  by rotating [ec, el et] around et.
            interp_mode (int): Choose interpolation factor:
                1: original method from Bayer et al.
                2: new method based on gradients.
                3: a better, more robust, implementation of the new method (default).
            linearize_u (boolean): When True, linearize longitudinal normalized coordinate with z-coordinate for z>0.
            mirror (boolean): When True, mirror the cardiac vectors above the equator to resolve boundary effects.
            retain_mode (str): 'et' or 'el' or 'combi'. Specify which unit vector (transmural or longitudinal)
                                should be retained/preserved when correcting for perpendicularity.
                                'Combi' option retains et near the z-axis and retains el elsewhere (recommended).
            transmural_coordinate_mode (int): Choose transmural coordinate method:
                1: Original (non-linear) transmural coordinate as used by Bayer et al 2012.
                2: Adapted method (recommended).
        """

        prm = Parameters('bayer')

        prm.add('apex_bc_mode', 'line')
        prm.add('correct_ec', True)
        prm.add('interp_mode', 3)
        prm.add('linearize_u', True)
        prm.add('mirror', True)
        prm.add('retain_mode', 'combi')
        prm.add('transmural_coordinate_mode', 2)

        # Verbose parameters for saving intermediate resuls to XDMF files.
        prm.add('verbose', False)  # If False, no XDMF files are saved.
        prm.add('dir_out', 'output/bayer')  # If verbose is True, specify output directory.

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this geometry object.
        """
        return self._parameters

    # Cardiac vectors.
    def ec(self, V):
        if not self._ec:
            if self._geometry_type == 'lv':
                self._create_vectors_lv(V)
            else:
                self._create_vectors_biv(V)
        return self._ec

    def el(self, V):
        if not self._el:
            if self._geometry_type == 'lv':
                self._create_vectors_lv(V)
            else:
                self._create_vectors_biv(V)
        return self._el

    def et(self, V):
        if not self._et:
            if self._geometry_type == 'lv':
                self._create_vectors_lv(V)
            else:
                self._create_vectors_biv(V)
        return self._et

    # Fiber vectors.
    def ef(self, V):
        if not self._ef:
            if self._geometry_type == 'lv':
                self._create_vectors_lv(V)
            else:
                self._create_vectors_biv(V)
        return self._ef

    def es(self, V):
        if not self._es:
            if self._geometry_type == 'lv':
                self._create_vectors_lv(V)
            else:
                self._create_vectors_biv(V)
        return self._es

    def en(self, V):
        if not self._en:
            if self._geometry_type == 'lv':
                self._create_vectors_lv(V)
            else:
                self._create_vectors_biv(V)
        return self._en

    # Wall bounded coordinates.
    def v_wall(self, V=None, *args):
        if self._v_wall is None:
            self._create_wall_bounded_coords(V, *args) # Is called in _create_vectors_biv
        return self._v_wall

    def v_septum(self, V=None, *args):
        if self._v_septum is None:
            self._create_wall_bounded_coords(V, *args) # Is called in _create_vectors_biv
        return self._v_septum

    def u(self, V=None, *args):
        if self._u is None:
            self._create_wall_bounded_coords(V, *args) # Is called in _create_vectors_biv
        return self._u

    # Interpolation factors.
    def t_endo(self, V=None):
        if self._t_endo is None:
            if self._geometry_type == 'lv':
                self._create_vectors_lv(V)
            else:
                self._create_vectors_biv(V)
        return self._t_endo

    def t_interp(self, V=None):
        if self._t_interp is None:
            if self._geometry_type == 'lv':
                self._create_vectors_lv(V)
            else:
                self._create_vectors_biv(V)
        return self._t_interp

    def check_geometry_type(self):
        if hasattr(self._geometry, 'rv_endocardium'):
            return 'biv'
        else:
            return 'lv'

    @staticmethod
    def laplace(V, geometry, boundary_1, boundary_2, apex_bc_mode='line'):
        """
        Solves the Laplace equation for the mesh with:
            laplace(u) = 0 in omega
            u = 1 on boundary_1
            u = 0 on boundary_2
            inner(gradient(u), n) = 0 on remaining boundary
        Args:
            V: The ufl vector function space of the displacement field of the FE simulation.
            geometry: A geometry object with attributes .mesh(), .tags() [mesh, facet function with boundary markers]
            boundary_1: boundary marker where the laplace function should be 1. Can be a list to combine domains.
            boundary_2: boundary marker where the laplace function should be 0. Can be a list to combine domains.
                        If boundary_2 = 'apex', then the lowest point of the mesh is taken as boundary_2.
            apex_bc_mode: Either 'line' or 'point'. If line, the BC for the apex are set to all the vertices of each
                          element that intersects the z-axis. If point, the apex BC is assigned to the lowest point in the mesh.
        Returns:
            u: Function with solution to Laplace problem.
        """

        # Extract function space and boundary markers.
        # TODO Check that V is in proper space.
        Q = vector_space_to_scalar_space(V)

        boundary_markers = geometry.tags()

        # Define Dirichlet boundary conditions.
        bcs = []
        if type(boundary_1) is not list:
            boundary_1 = [boundary_1]
        for b in boundary_1:
            bc_1 = DirichletBC(Q, Constant(1.), boundary_markers, b)
            bcs.append(bc_1)

        if boundary_2 == 'apex':
            if apex_bc_mode == 'line':
                # Find the cells that intersect the z-asix.
                tree = geometry.mesh().bounding_box_tree()
                tree.build(geometry.mesh(), 3)
                num_cells = geometry.mesh().num_cells()
                try:
                    prm = geometry.parameters['geometry']
                    zmin = -prm['Z_2']
                    zmax = -prm['Z_1']
                except KeyError:
                    X = Q.tabulate_dof_coordinates().reshape(-1, 3)
                    zmin_ = min(X[:,2])
                    zmin = MPI.min(mpi_comm_world(), zmin_)
                    zmax = zmin/2
                stepsize = geometry.mesh().rmin()
                idx_apex = []
                cells_apex = []
                for z in np.arange(zmin, zmax, stepsize):
                    point_apex = Point(0., 0., z)
                    # Get the indices of the cells that intersects with the apex point.
                    idx_all = tree.compute_collisions(point_apex)
                    for idx in idx_all:
                        if idx < num_cells and idx not in idx_apex:
                            # A new apex cell is found.
                            idx_apex.append(idx)
                            # Get the cell.
                            cells_apex.append(Cell(geometry.mesh(), idx))

                def apex_point(x, on_boundary):
                    # Check whether the point intersects with any of the apex cells.
                    for cell in cells_apex:
                        if cell.collides(Point(x)):
                            return True
                    return False

                bc_apex = Expression('sqrt(x[0]*x[0] + x[1]*x[1])', degree=3)
                bc_2 = DirichletBC(Q, bc_apex, apex_point, method='pointwise')

            elif apex_bc_mode == 'point':
                # Find lowest point in the mesh.
                X = Q.tabulate_dof_coordinates().reshape(-1, 3)
                zmin_ = min(X[:,2])
                zmin = MPI.min(mpi_comm_world(), zmin_)

                def apex_point(x, on_boundary):
                    # Check whether the point is the lowest point in the mesh.
                    return abs(x[2] - zmin) < 1e-20

                bc_2 = DirichletBC(Q, Constant(0.0), apex_point, method='pointwise')

            else:
                raise ValueError('Unknown apex_bc_mode "{}"'.format(apex_bc_mode))

            # Append to bcs.
            bcs.append(bc_2)

        else:
            if type(boundary_2) is not list:
                boundary_2 = [boundary_2]
            for b in boundary_2:
                bc_2 = DirichletBC(Q, Constant(0.), boundary_markers, b)
                bcs.append(bc_2)

        # Define trial and test functions.
        u = TrialFunction(Q)
        v = TestFunction(Q)

        # Define variational problem.
        f = Constant(0.0)
        a = dot(grad(u), grad(v)) * dx
        L = f * v * dx

        u = Function(Q)
        solve(a == L, u, bcs)

        return u

    @staticmethod
    def axis(grad_psi, grad_phi, retain, correct_ec=True):
        """
        Given the gradient of the solution of the laplace function from endocardium to epicardium (grad_phi)
        and from base to apex (grad_psi), construct a 3x3 orthogonal basis consisting of
        longitudinal (el), circumferential (ec) and transmural (et) unit vectors.

        Args:
            grad_psi (1x3 numpy array): the gradient of psi at one point
                                        (which is the solution of the Laplace equation in longitudinal direction).
            grad_phi (1x3 numpy array): the gradient of phi at one point
                                        (which is the solution of the Laplace equation in transmural direction).
            retain (str): Choose between 'el' (default) or 'et'.
                          Specifies which unit vector (longitudinal or transmural) is preserved/retained
                          when enforcing perpendicularity between el and et.
            correct_ec (boolean): Whether to enforce the z-component of the circumferential direction to be zero
                                  by rotating around et.
        Returns:
            tuple with the unit vectors for ec, el and et.
        """

        if retain == 'el':
            # Longitudinal
            el = grad_psi / norm(grad_psi)

            # Transmural (force perpendicularity, retaining original el)
            tm = grad_phi - np.inner(el, grad_phi) * el
            et = tm / norm(tm)

        elif retain == 'et':
            # Transmural
            et = grad_phi / norm(grad_phi)

            # Longitudinal (force perpendicularity, retaining original et)
            lo = grad_psi - np.inner(et, grad_psi) * et
            el = lo / norm(lo)
        else:
            raise NotImplementedError('Given argument for retain in CardiacBasisVectorBayer.axis "{}" not implemented. Choose "et" or "el".'.format(retain))

        # Circumferential
        ec = np.cross(el, et)

        if correct_ec:

            # Compute angle with which to rotate around et to get ec in xy-plane.
            gamma = np.arctan2(-ec[2], el[2])

            # Roatation matrix.
            R = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                          [np.sin(gamma),  np.cos(gamma), 0],
                          [0,              0,             1]])

            # Rotate around et.
            ec, el, et = np.transpose(np.transpose(np.array((ec, el, et))) @ R)

        return ec, el, et

    @staticmethod
    def project_gradient(u, V, quadrature_degree=None):
        """
        Project grad(u) to the vector function space V. Should be similar (but faster) to using project(grad(u), V).
        https://fenicsproject.org/qa/1425/derivatives-at-the-quadrature-points/

        Args:
            u: Scalar function
            V: VectorFunctionSpace
            quadrature_degree (optional): Quadrature degree
        Returns:
            A FEniCS VectorFunction
        """

        # You want to solve this variational form
        # inner(uq, vq)*dx = inner(vq, grad(u))*dx
        uq = TrialFunction(V)
        vq = TestFunction(V)

        # The mass matrix is diagonal. Get the diagonal
        if quadrature_degree is None:
            M = assemble(inner(uq, vq) * dx)
        else:
            M = assemble(inner(uq, vq) * dx, form_compiler_parameters = {'quadrature_degree': quadrature_degree})
        ones = Function(V)
        ones.vector()[:] = 1.
        Md = M * ones.vector()

        # If you only want to assemble right hand side once:
        if quadrature_degree is None:
            grad_eps = assemble(inner(vq, grad(u)) * dx)
        else:
            grad_eps = assemble(inner(vq, grad(u)) * dx, form_compiler_parameters = {'quadrature_degree': quadrature_degree})

        # solve
        v1 = Function(V)
        v1.vector().set_local(grad_eps.array() / Md.array())

        return v1

    @staticmethod
    def project_gradient_local(u, V):
        """
        Uses a local (faster) solver to compute gradient u in function space V.
        Args:
            u: Scalar function
            V: FunctionSpace
        Returns:
            A (vector) Function
        """
        w = TrialFunction(V)
        v = TestFunction(V)

        a = inner(w, v) * dx
        L = inner(grad(u), v) * dx

        gradu = Function(V)
        ls = LocalSolver(a, L)
        ls.solve_local_rhs(gradu)

        return gradu

    @staticmethod
    def orient(ax, ah, at):
        """
        Args:
            ax: a tuple with 3 orthogonal basis vectors: local cardiac vector basis: (ec, el, et).
            ah: fiber (helix) angle.
            at: transverse angle.
        Returns:
            a tuple with fiber basis vectors (ef, en, es)
        """
        # Collect unit vectors in orthogonal matrix Q.
        Q = np.zeros((3, 3))
        Q[:,0] = ax[0]
        Q[:,1] = ax[1]
        Q[:,2] = ax[2]

        # Calculate gamma (transverse angle in ec'-et plane (after rotation with helix angle ah)).
        f_c = 1/np.sqrt((np.tan(ah))**2 + (np.tan(at))**2 + 1)
        f_l = np.tan(ah) * f_c
        f_t = np.tan(at) * f_c
        gamma = np.arctan2(f_t, np.sqrt(f_c**2 + f_l**2))

        # Create rotation matrices.
        # First rotation: rotate around et with angle ah.
        R_ah = np.array([[np.cos(ah), -np.sin(ah), 0],
                        [np.sin(ah), np.cos(ah), 0],
                        [0, 0, 1]])

        # Second rotation: rotate around el' with angle -gamma (the tranverse angle in ec'-el plane).
        R_at = np.array([[np.cos(-gamma), 0, np.sin(-gamma)],
                        [0, 1, 0],
                        [-np.sin(-gamma), 0, np.cos(-gamma)]])

        # Rotate Q.
        Q_ = Q @ R_ah @ R_at
        return Q_[:,0], Q_[:,1], Q_[:,2]

    @staticmethod
    def bislerp(ax1, ax2, t, flip='auto'):
        """
        Bilinear spherical interpolation as proposed by Bayer et al. 2012.
        Args:
            ax1: tuple with 3 3d orthogonal unit vectors (as returned by the axis function).
                 This axis will be rotated 180 degrees to find an orientation closest to ax2.
            ax2: tuple with 3 3d orthogonal unit vectors (as returned by the axis function).
            t: interpolation parameter [0,1] for interpolating between basis ax1 and ax2.
            flip: If True, flips first and last axis in ax1. If False does not flip at all.
                  If 'auto', chooses to flip based on angle between ax1 and ax2.
        Returns:
            Tuple with 3 unit vectors representing the interpolated basis.
        """

        # Flip the et vector of the first axis and check whether this lies closer to the et vector in the second axis
        # since we want to interpolate the smallest angle possible and fiber direction does not matter. We do not care
        # about the longitudinal vectors since they will be oriented in the same way (pointing upwards) for both axis.
        # Note that this is the only difference between slerp and bislerp.
        did_we_flip = False
        if flip == 'auto':
            if np.inner(-ax1[2], ax2[2]) > np.inner(ax1[2], ax2[2]):
                ax1 = (-ax1[0], ax1[1], -ax1[2]) # Flip 2 axis to keep a right-handed system.
                did_we_flip = True
        elif flip is True:
            ax1 = (-ax1[0], ax1[1], -ax1[2])  # Flip 2 axis to keep a right-handed system.

        # Convert the basis to orthogonal matrices.
        Q_A = np.zeros((3, 3))
        Q_A[:, 0] = ax1[0]  # circumferential
        Q_A[:, 1] = ax1[1]  # longitudinal
        Q_A[:, 2] = ax1[2]  # transmural

        Q_B = np.zeros((3, 3))
        Q_B[:, 0] = ax2[0]  # circumferential
        Q_B[:, 1] = ax2[1]  # longitudinal
        Q_B[:, 2] = ax2[2]  # transmural

        # Convert to quaternions.
        q_M = QuatMath.rot2quat(Q_A)
        q_B = QuatMath.rot2quat(Q_B)

        q_int = QuatMath.slerp(q_M, q_B, t)

        Q_int = QuatMath.quat2rot(q_int)

        return (Q_int[:, 0], Q_int[:, 1], Q_int[:, 2]), did_we_flip

    @staticmethod
    def lagrange_to_quadrature(V_q, *args):
        """
        Interpolates nodal values of functions in Lagrange space to nodal values in quadrature space.

        Args:
            V_q: Quadrature vector space (to interpolate the nodal values on).
            *args: 1D or 2D arrays with nodal values of 1D or 2D functions in Lagrange function spaces.
            The given quadrature vector space V_q will be converted to a scalar space automatically for 1D arrays.

        Returns:
            1D or 2D arrays with nodal values of 1D or 2D functions in quadrature space.
        """
        # Create quadratic Lagrange function spaces.
        mesh = V_q.ufl_domain().ufl_cargo()
        V = VectorFunctionSpace(mesh, 'Lagrange', 2)
        Q = FunctionSpace(mesh, 'Lagrange', 2)

        # Create scalar Quadrature function space.
        Q_q = vector_space_to_scalar_space(V_q)

        # Create Lagrangian functions.
        f_V = Function(V)
        f_Q = Function(Q)

        out = ()
        for array in args:
            if array is not None:
                # Get dimension of array.
                dim = array.ndim

                # Assign the array to a Lagrangian function and interpolate to quadrature space.
                if dim == 1:
                    reset_values(f_Q, array)
                    f_quad = interpolate(f_Q, Q_q)
                    quad_array = f_quad.vector().array()
                elif dim==2:
                    # TODO validate this implementation in parallel.
                    reset_values(f_V, array.reshape(-1))
                    f_quad = interpolate(f_V, V_q)
                    quad_array = f_quad.vector().array().reshape([-1, array.shape[1]])
                else:
                    raise NotImplementedError
            else:
                quad_array = None

            # Return the nodal values and coordinates of the quadrature space.
            out += (quad_array,)
        return out

    def _create_vectors_lv(self, V):
        """
        Cardiac basic vectors according tp the Bayer method for an LV mesh.
        Returns components of unit vectors on discretized function space V.

        NOTE this one in not updated. It does not create the fiber vectors, only cardiac vectors.
        Also, the mirror option is not implemented for the LV model.

        Args:
            V: Function space to define the fiber vectors on.
       """
        # Check if V is a quadrature function space.
        if V.ufl_element().family() == 'Quadrature':
            # Save quadrature space as V_q.
            V_q = V

            # Create a quadratic Lagrange space to solve the Laplace equations on.
            mesh = V_q.ufl_domain().ufl_cargo()
            V = VectorFunctionSpace(mesh, 'Lagrange', 2)
        else:
            V_q = None

        # Solve energy potential function phi from epicardium to endocardium.
        phi = self.laplace(V, self._geometry, self._geometry.epicardium, self._geometry.endocardium)

        # Solve energy potential function psi from base to apex.
        if hasattr(self._geometry, 'apex'):
            apex_marker = self._geometry.apex
        else:
            apex_marker = 'apex'
        psi = self.laplace(V, self._geometry, self._geometry.base, apex_marker, self.parameters['apex_bc_mode'])

        # Project grad(psi) and grad(phi) to the function space.
        grad_psi_ = self.project_gradient(psi, V)
        grad_phi_ = self.project_gradient(phi, V)

        grad_psi = grad_psi_.vector().array().reshape(-1, 3)
        grad_phi = grad_phi_.vector().array().reshape(-1, 3)

        ec = np.zeros(np.shape(grad_psi))
        el = np.zeros(np.shape(grad_psi))
        et = np.zeros(np.shape(grad_psi))

        if self.parameters['retain_mode'] == 'combi':
            # Choose retain mode based on distance to the z-axis.
            m_cells = [Cell(self._geometry.mesh(), idx) for idx in range(self._geometry.mesh().num_cells())]
            # Compute the radius of each cell in the mesh.
            radii = [each.circumradius() * 0.75 for each in m_cells]
            mean_radius = np.mean(radii)
            tol = (2.5*mean_radius)**2

        # Check if we need to convert the solutions to the quadrature space.
        if V_q is None:
            # Do not interpolate to quadrature space.
            Q = vector_space_to_scalar_space(V)
            X = Q.tabulate_dof_coordinates().reshape(-1, 3)
        else:
            # Interpolate to quadrature space.
            Q_q = vector_space_to_scalar_space(V_q)
            X = Q_q.tabulate_dof_coordinates().reshape(-1, 3)
            grad_psi, grad_phi = self.lagrange_to_quadrature(V_q, grad_psi, grad_phi)

        for ii in range(len(grad_psi)):
            if self.parameters['retain_mode'] == 'combi':
                if (X[ii,0]**2 + X[ii,1]**2) <= tol:
                    retain = 'et'
                else:
                    retain = 'el'
            else:
                retain=self.parameters['retain_mode']
            ec[ii, :], el[ii, :], et[ii, :] = self.axis(grad_psi[ii, :], grad_phi[ii, :], retain=retain)

        # Store the components as tuples to be compatible with CoordinateSystem.
        self._ec = (ec[:, 0], ec[:, 1], ec[:, 2])
        self._el = (el[:, 0], el[:, 1], el[:, 2])
        self._et = (et[:, 0], et[:, 1], et[:, 2])

    def _create_wall_bounded_coords(self, V, phi_rv, phi_lv, phi_epi, grad_phi_rv, grad_phi_lv, psi_func):
        """
        Creates normalized wall bounded transmural coordinates v based on Bayer et al 2012,
        but with modifications to obtain (nearly) linear wall bound coordinates.
        Args:
            :param V: UFL function space
            :param phi_rv: numpy array with nodal values of the Laplace solution for RV (see Bayer et al. 2012)
            :param phi_lv: numpy array with nodal values of the Laplace solution for LV
            :param phi_epi: numpy array with nodal values of the Laplace solution for epicardium
            :param grad_phi_rv: numpy array with nodal values of the gradient of the Laplace solution for RV
            :param grad_phi_lv: numpy array with nodal values of the gradient of the Laplace solution for LV
            :param psi_func: FEniCS function of Laplace solution from base to apex.
        Returns:
            :return: None, but fills the attributes self._v_wall and self._v_septum with nodal values of the
                     transmural wall bounded coordinate.
        """
        # Collect parameters for defining wall bound coordinates (transmural).
        geo_pars = self._geometry.parameters['geometry']
        R_1 = geo_pars['R_1']
        R_2 = geo_pars['R_2']
        R_3 = geo_pars['R_3']
        R_4 = geo_pars['R_4']
        R_3y = geo_pars['R_3y']
        R_1sep = geo_pars['R_1sep']
        R_2sep = geo_pars['R_2sep']
        Z_1 = geo_pars['Z_1']
        Z_2 = geo_pars['Z_2']
        Z_3 = geo_pars['Z_3']
        Z_4 = geo_pars['Z_4']
        h = geo_pars['h']

        verbose = self.parameters['verbose']
        dir_out = self.parameters['dir_out']

        # Focus LV free wall.
        C_lv = np.sqrt(Z_1 ** 2 - R_1 ** 2)
        # Inner and outer sigma for LV free wall (compute sig at (0, 0, Z_1) and (0, 0, Z_2)).
        sig_inner = Z_1 / C_lv
        sig_outer = Z_2 / C_lv

        # Extract the coordinates of the degrees of freedom.
        Q = vector_space_to_scalar_space(V)
        X = Q.tabulate_dof_coordinates().reshape(-1, 3)

        # ------------------------------------------------------------
        # Longitudinal coordinate u: interpolate relation between psi and analytical u at LV free wall.
        # ------------------------------------------------------------
        # Choose sigma and tau.
        n = 400  # number of sample points.
        phi_sample = np.zeros(n) + 0  # y=0
        sig_sample = np.zeros(n) + (sig_inner + sig_outer) / 2  # Halfway through the wall
        tau_max = h / C_lv / sig_sample[0]
        # Create a parabolic function tau(x) where tau(-1) = -1, tau(tau_max) = tau_max, and with the minimum at x=-1.
        # This ensures that there are more points taken at the apex than if we would use a linear tau array.
        x = np.linspace(-1, tau_max, n)
        a = (-1 - tau_max) / (1 - tau_max ** 2 - 2 - 2 * tau_max)
        b = 2 * a
        c = b - a - 1
        tau_sample = a * x ** 2 + b * x + c  # from apex to base

        # Compute Cartesian coordinates.
        X_sample = GeoFunc.ellipsoidal_to_cartesian(sig_sample, tau_sample, phi_sample, C_lv)

        # Compute normalized longitudinal coordinate (u).
        u_sample = GeoFunc.analytical_u(X_sample, sig_sample, tau_sample, h, linearlize=self.parameters['linearize_u'])

        # Compute psi at sample points.
        psi_sample = np.zeros(len(X_sample))
        for ii in range(len(X_sample)):
            try:
                psi_sample[ii] = psi_func(X_sample[ii, :])
            except:
                psi_sample[ii] = -1

            # Gather the evaluations (some processes may not contain the point).
            psi_sample[ii] = MPI.max(mpi_comm_world(), psi_sample[ii])

        idx_include = psi_sample > -1
        psi_sample = psi_sample[idx_include]
        u_sample = u_sample[idx_include]

        # Interpolate.
        psi_to_u = scipy.interpolate.interp1d(psi_sample, u_sample, kind='quadratic',
                                        fill_value=(-1, max(u_sample)), bounds_error=False)

        # Store nodal u values.
        u = psi_to_u(psi_func.vector().array())
        self._u = u

        # ------------------------------------------------------------
        # Transmural coordinate v
        # ------------------------------------------------------------
        v_septum = np.zeros(np.shape(phi_rv))
        v_wall = np.zeros(np.shape(phi_rv))
        for ii in range(len(phi_rv)):
            # v_septum
            if phi_rv[ii] < 1e-15 and phi_lv[ii] < 1e-15:  # At boundary
                v_septum_ii = np.linalg.norm(grad_phi_rv[ii, :]) / (
                        np.linalg.norm(grad_phi_lv[ii, :]) + np.linalg.norm(grad_phi_rv[ii, :]))
            else:
                v_septum_ii = phi_rv[ii] / (phi_lv[ii] + phi_rv[ii])

            # v_wall
            if self.parameters['transmural_coordinate_mode'] == 2:  # New method
                # Free wall.
                x, y, z = X[ii, :]
                if x >= 0:  # LV free wall: analytical solution.
                    # Compute sig and tau.
                    sig = GeoFunc.compute_sigma(x, y, z, C_lv)
                    tau = GeoFunc.compute_tau(x, y, z, C_lv)
                    # Compute normalized radial coordinate v.
                    v_wall_ii = GeoFunc.analytical_v(sig, tau, sig_inner, sig_outer)
                else:
                    # Check if we are at RV or Septum.
                    if x < GeoFunc.compute_x_sep(y, z, R_2, R_2sep, Z_2):  # RV free wall.
                        a_endo = R_3
                        a_epi = R_4
                        b_endo = R_3y
                        b_epi = R_2
                        c_endo = Z_3
                        c_epi = Z_4
                        # Choose angles smaller than for septum since the RV is further away from the origin,
                        # meaning that a certain dphi is a larger ds on the RV surface.
                        d_phi = 0.001
                        d_theta = 0.001
                    else:  # Septum
                        a_endo = R_1sep
                        a_epi = R_2sep
                        b_endo = R_1
                        b_epi = R_2
                        c_endo = Z_1
                        c_epi = Z_2
                        d_phi = 0.01
                        d_theta = 0.01
                    # Find closest point to endocard and epicard and calculate transmural depth.
                    v_wall_ii = GeoFunc.closest_distance_v(x, y, z, a_endo, a_epi, b_endo, b_epi, c_endo, c_epi,
                                                           d_phi=d_phi, d_theta=d_theta)

            elif self.parameters['transmural_coordinate_mode'] == 1:  # Original method (Bayer et al.)
                v_wall_ii = phi_epi[ii]
            else:
                raise ValueError('Unknown transmural_coordinate_mode.')

            v_septum[ii] = v_septum_ii
            v_wall[ii] = v_wall_ii

        if self.parameters['transmural_coordinate_mode'] == 2:  # New method.
            # Convert the septal wallbounded coordinates to a FEniCS function.
            v_septum_func = Function(vector_space_to_scalar_space(V))
            v_septum_func.vector()[:] = v_septum
            v_septum_func.vector().apply('')

            # Correct the septal transmural coordinate (linearize it).
            v_septum_func = GeoFunc.interpolate_v_septum(self._geometry, v_septum_func)

            # Save the correct septal coordinates as an array with the nodal values.
            v_septum = v_septum_func.vector().array()

            if verbose:
                f = self.array_to_function(v_wall, V, name='v_wall_semi_anal')
                save_to_disk(f, os.path.join(dir_out, 'v_wall_semi_anal.xdmf'))

            # Take minimum of phi_epi and v_wall as final v_wall.
            v_wall = np.min((v_wall, phi_epi), axis=0)

        # Convert to domain [-1, 1].
        v_wall = 2 * v_wall - 1
        v_septum = 2 * v_septum - 1

        self._v_wall = v_wall
        self._v_septum = v_septum

    def _create_vectors_biv(self, V):
        """
        Cardiac basic vectors according to the Bayer method for a biventricular mesh.
        Returns components of unit vectors on discretized function space V.

        Args:
            V: Function space to define the fiber vectors on.
        """
        # Extract paramaters.
        verbose = self.parameters['verbose']
        dir_out = self.parameters['dir_out']

        # Check if V is a quadrature function space.
        if V.ufl_element().family() == 'Quadrature':
            # Save quadrature space as V_q.
            V_q = V

            # Create a quadratic Lagrange space to solve the Laplace equations on.
            mesh = V_q.ufl_domain().ufl_cargo()
            V = VectorFunctionSpace(mesh, 'Lagrange', 2)
        else:
            V_q = None

        # Collect boundary markers.
        boundary_epi = [self._geometry.lv_epicardium, self._geometry.rv_epicardium]
        boundary_lv = [self._geometry.lv_endocardium]
        boundary_rv = [self._geometry.rv_endocardium, self._geometry.rv_septum]

        # Solve energy potential function phi from epicardium to endocardium.
        phi_epi_ = self.laplace(V,
                                self._geometry,
                                boundary_epi,
                                boundary_lv + boundary_rv)

        phi_lv_ = self.laplace(V,
                               self._geometry,
                               boundary_lv,
                               boundary_epi + boundary_rv)

        # Solve energy potential function psi from base to apex.
        if hasattr(self._geometry, 'apex'):
            boundary_apex = self._geometry.apex
        else:
            boundary_apex = 'apex'
        psi_ = self.laplace(V,
                           self._geometry,
                           self._geometry.base,
                           boundary_apex,
                           self.parameters['apex_bc_mode'])

        # Project grad(psi) and grad(phi) to the function space.
        grad_phi_epi_ = self.project_gradient(phi_epi_, V) # project(grad(phi_epi_), V)
        grad_phi_lv_ = self.project_gradient(phi_lv_, V) # project(grad(phi_lv_), V)
        grad_psi_ = self.project_gradient(psi_, V) # project(grad(psi_), V)

        # Extract the coordinates of the degrees of freedom.
        Q = vector_space_to_scalar_space(V)
        X = Q.tabulate_dof_coordinates().reshape(-1, 3)

        # Convert to array form.
        if self.parameters['mirror']:
            # Mirror if requested.
            # Note: self.mirror takes functions, returns arrays
            grad_phi_epi, grad_phi_lv, grad_psi = self.mirror(X, grad_phi_epi_, grad_phi_lv_, grad_psi_)
        else:
            # Do not mirror, simply convert to arrays.
            grad_phi_epi = grad_phi_epi_.vector().array().reshape(-1, 3)
            grad_phi_lv = grad_phi_lv_.vector().array().reshape(-1, 3)
            grad_psi = grad_psi_.vector().array().reshape(-1, 3)

        # Convert functions to arrays (values at dofs).
        phi_epi = phi_epi_.vector().array()
        phi_lv = phi_lv_.vector().array()
        phi_rv = 1 - phi_lv - phi_epi  # Make use of conservation phi_rv + phi_lv + phi_epi = 1

        # Make use of conservation grad_phi_rv + grad_phi_lv + grad_phi_epi = 0
        grad_phi_rv = -grad_phi_lv - grad_phi_epi

        # Wall bounded coordinates.
        v_wall = self.v_wall(V, phi_rv, phi_lv, phi_epi, grad_phi_rv, grad_phi_lv, psi_)
        v_septum = self.v_septum(V, phi_rv, phi_lv, phi_epi, grad_phi_rv, grad_phi_lv, psi_)
        u = self.u(V, phi_rv, phi_lv, phi_epi, grad_phi_rv, grad_phi_lv, psi_)

        # If we use the modified interpolation factor/map, we need this.
        if self.parameters['interp_mode'] == 2:
            phi_endo = Function(Q)
            phi_endo.vector()[:] = phi_rv/(phi_rv+phi_lv+1e-12)
            phi_endo.vector().apply('')
            grad_phi_endo = self.project_gradient(phi_endo, V).vector().array().reshape(-1, 3)

        elif self.parameters['interp_mode'] == 3:
            # This needs a high number of quadrature points (yields a different and better result than mode 2!)
            phi_rv_ = 1 - phi_lv_ - phi_epi_
            phi_endo = phi_rv_/(phi_rv_+phi_lv_)
            grad_phi_endo = self.project_gradient(phi_endo, V, quadrature_degree=7).vector().array().reshape(-1, 3)
        else:
            grad_phi_endo = None

        # Check if we need to convert the solutions to the quadrature space.
        if V_q is not None:
            # Interpolate to quadrature space.
            Q_q = vector_space_to_scalar_space(V_q)
            X = Q_q.tabulate_dof_coordinates().reshape(-1, 3)
            grad_phi_endo, grad_phi_epi, grad_phi_lv, grad_phi_rv, grad_psi, phi_epi, phi_lv, phi_rv, v_septum, v_wall, u = self.\
                lagrange_to_quadrature(V_q, grad_phi_endo, grad_phi_epi, grad_phi_lv, grad_phi_rv, grad_psi, phi_epi, phi_lv, phi_rv, v_septum, v_wall, u)

        # Choose retain mode based on distance to the z-axis (optional).
        m_cells = [Cell(self._geometry.mesh(), idx) for idx in range(self._geometry.mesh().num_cells())]
        # Compute the radius of each cell in the mesh.
        radii = [each.circumradius() * 0.75 for each in m_cells]
        mean_radius = np.mean(radii)

        # Prepare for loop.
        ec = np.zeros(np.shape(grad_psi))
        el = np.zeros(np.shape(grad_psi))
        et = np.zeros(np.shape(grad_psi))
        en = np.zeros(np.shape(grad_psi))
        ef = np.zeros(np.shape(grad_psi))
        es = np.zeros(np.shape(grad_psi))
        et_lv_array = np.zeros(np.shape(grad_psi))
        et_rv_array = np.zeros(np.shape(grad_psi))
        et_epi_array = np.zeros(np.shape(grad_psi))
        et_endo_array = np.zeros(np.shape(grad_psi))
        ef_endo_array = np.zeros(np.shape(grad_psi))
        ef_epi_array = np.zeros(np.shape(grad_psi))
        t_endo = np.zeros(len(grad_psi))
        t_interp = np.zeros(len(grad_psi))
        retain_array = np.empty(len(grad_psi))
        retain_array[:] = np.nan

        # For every dof, determine local cardiac basis vectors.
        for ii in range(len(grad_psi)):

            grad_psi_ii = grad_psi[ii, :]

            # Determine retain mode.
            if self.parameters['retain_mode'] == 'combi':
                if (X[ii,0]**2 + X[ii,1]**2) <= (2.5*mean_radius)**2:
                    retain = 'et'
                    retain_array[ii] = 0
                else:
                    retain = 'el'
                    retain_array[ii] = 1
            else:
                retain=self.parameters['retain_mode']

            # Determine interpolation factor for interpolating Qlv and Qrv.
            # Difference with Bayer: at boundary use the magnitudes of the gradient to determine the
            # interpolation factor (to avoid problems at the boundary where phi_rv=phi_lv=0).
            if phi_lv[ii] < 1e-15 and phi_rv[ii] < 1e-15: # At boundary
                t_endo_ii = norm(grad_phi_rv[ii, :])/(norm(grad_phi_lv[ii, :]) + norm(grad_phi_rv[ii, :]))
            else:
                t_endo_ii = phi_rv[ii] / (phi_lv[ii] + phi_rv[ii])

            # Interpolation factor for interpolating Qendo and Qepi.
            if self.parameters['interp_mode'] == 1: # Original
                t_interp_ii = phi_epi[ii]
            elif self.parameters['interp_mode'] == 2 or self.parameters['interp_mode'] == 3: # New method
                t_interp_ii = np.max((norm(grad_phi_epi[ii, :]) / (norm(grad_phi_epi[ii, :]) + norm(grad_phi_endo[ii, :])), phi_epi[ii]))
            else:
                raise NotImplementedError('Unknown interpolation method.')

            # Longitudinal coordinate u.
            u_ii = u[ii]

            # Transmural coordinate v.
            v_lv = v_septum[ii]
            v_rv = v_septum[ii]
            v_epi = v_wall[ii]

            # Calculate helix angles.
            ah_lv = self._func_ah_s(u_ii, v_lv)
            ah_rv = self._func_ah_s(u_ii, v_rv)
            ah_epi = self._func_ah_w(u_ii, v_epi)

            # Calculate transverse angles.
            if self._func_at is not None:
                at_lv = self._func_at(u_ii, v_lv)
                at_rv = self._func_at(u_ii, v_rv)
                at_epi = self._func_at(u_ii, v_epi)
            else:
                at_lv, at_rv, at_epi = 0, 0, 0

            # Cardiac vectors for septal parts.
            Q_lv = self.axis(grad_psi_ii, -grad_phi_lv[ii, :], retain=retain, correct_ec=self.parameters['correct_ec'])
            Q_rv = self.axis(grad_psi_ii,  grad_phi_rv[ii, :], retain=retain, correct_ec=self.parameters['correct_ec'])

            et_lv = Q_lv[2]
            et_rv = Q_rv[2]

            # Cardiac vectors for epicardium parts.
            Q_epi = self.axis(grad_psi_ii, grad_phi_epi[ii, :], retain=retain, correct_ec=self.parameters['correct_ec'])
            et_epi = Q_epi[2]

            # Fiber vectors for septal parts.
            Q_lv_f = self.orient(Q_lv, ah=ah_lv, at=at_lv)
            Q_rv_f = self.orient(Q_rv, ah=ah_rv, at=at_rv)

            # Fiber vectors for epicardium parts.
            Q_epi_f = self.orient(Q_epi, ah=ah_epi, at=at_epi)
            ef_epi = Q_epi_f[0]

            # Interpolate Qlv and Qrv.
            Q_endo_f, flip = self.bislerp(Q_lv_f, Q_rv_f, t_endo_ii)
            Q_endo = self.bislerp(Q_lv, Q_rv, t_endo_ii, flip=flip)[0]
            et_endo = Q_endo[2]
            ef_endo = Q_endo_f[0]

            # Interpolate Qendo and Qepi.
            Q_interp_f, flip = self.bislerp(Q_endo_f, Q_epi_f, t_interp_ii)
            Q_interp = self.bislerp(Q_endo, Q_epi, t_interp_ii, flip=flip)[0]

            ec_ii, el_ii, et_ii = Q_interp
            ef_ii, en_ii, es_ii = Q_interp_f

            # Check if et should be flipped: circumferential vectors should run anti-clockwise.
            coords = X[ii, :]
            if (norm(coords[:2]) > mean_radius and norm((coords + np.cross([0,0,1], ec_ii))[:2]) > norm((coords + np.cross([0,0,1], -ec_ii))[:2]))\
                    or (norm(coords[:2]) <= mean_radius and norm(coords - et_ii) > norm(coords + et_ii)): # Special treatment for apex -> transmural vector should point away from origin. # or (coords[0]<0 and et_ii[0]>0.7):
                # Flip both et and ec (rotate 180 around el).
                ec_ii = -ec_ii
                et_ii = -et_ii

                # Also flip corresponding fiber vectors (rotate 180 around en)
                # (does not influence orientation really, just for visualization purposes).
                ef_ii = -ef_ii
                es_ii = -es_ii

            ec[ii, :], el[ii, :], et[ii, :] = ec_ii, el_ii, et_ii
            ef[ii, :], en[ii, :], es[ii, :] = ef_ii/norm(ef_ii), -en_ii/norm(en_ii), es_ii/norm(es_ii)  # Flip en to keep a righthanded system (en originates from the second unit vector of a right-handed system), in subsequent code, en is the third vector (ef, es, en).

            # Save interpolation factors for possible visualization.
            t_endo[ii] = t_endo_ii
            t_interp[ii] = t_interp_ii

            # Save intermediate vectors.
            et_lv_array[ii, :] = et_lv
            et_rv_array[ii, :] = et_rv
            et_endo_array[ii, :] = et_endo
            et_epi_array[ii, :] = et_epi
            ef_endo_array[ii, :] = ef_endo
            ef_epi_array[ii, :] = ef_epi

        # Store the components as tuples to be compatible with CoordinateSystem.
        self._ec = (ec[:, 0], ec[:, 1], ec[:, 2])
        self._el = (el[:, 0], el[:, 1], el[:, 2])
        self._et = (et[:, 0], et[:, 1], et[:, 2])

        self._ef = (ef[:, 0], ef[:, 1], ef[:, 2])
        self._en = (en[:, 0], en[:, 1], en[:, 2])
        self._es = (es[:, 0], es[:, 1], es[:, 2])

        # Store interpolation factors
        self._t_endo = t_endo
        self._t_interp = t_interp

        if verbose:
            # Save functions.
            save_to_disk(psi_, os.path.join(dir_out, 'psi.xdmf'))
            save_to_disk(phi_epi_, os.path.join(dir_out, 'phi_epi.xdmf'))
            save_to_disk(phi_lv_, os.path.join(dir_out, 'phi_lv.xdmf'))
            phi_rv_ = project(1 - phi_lv_ - phi_epi_, Q)
            save_to_disk(phi_rv_, os.path.join(dir_out, 'phi_rv.xdmf'))

            # Convert arrays to functions and save.
            save_arrays_dict = {
                'grad_psi': grad_psi,
                'grad_phi_epi': grad_phi_epi,
                'grad_phi_lv': grad_phi_lv,
                'grad_phi_rv': grad_phi_rv,

                'v_wall': v_wall,
                'v_septum': v_septum,
                'u': u,

                't_endo': t_endo,
                't_interp': t_interp,

                'retain': retain_array,

                'et_lv': et_lv_array,
                'et_rv': et_rv_array,
                'et_endo': et_endo_array,
                'et_epi': et_epi_array,
                'ef_epi': ef_epi_array,
                'ef_endo': ef_endo_array}

            for name in save_arrays_dict.keys():
                array = save_arrays_dict[name]
                f = self.array_to_function(array, V, V_q=V_q, name=name)
                save_to_disk(f, os.path.join(dir_out, name+'.xdmf'))

    @staticmethod
    def array_to_function(array, V, V_q=None, name=''):

        # Get dimension of array.
        dim = array.ndim

        if V_q is not None:
            # Array contains values of quadrature space.
            # Check whether it is a scalar space or vector space.
            if dim == 1:
                # Scalar space.
                S = vector_space_to_scalar_space(V)
                f_q = Function(vector_space_to_scalar_space(V_q))

            elif dim == 2:
                # Vector space.
                S = V
                f_q = Function(V_q)

            else:
                raise NotImplementedError

            # Create quadrature function.
            reset_values(f_q, array.reshape(-1))

            # Create Function for projection.
            f = Function(S, name=name)

            # Project onto function space V.
            project(f_q, S, function=f)

        else:
            # Array contains values of reguar function space.
            # Check whether it is a scalar space or vector space.
            if dim == 1:
                # Scalar space.
                f = Function(vector_space_to_scalar_space(V), name=name)

            elif dim == 2:
                # Vector space.
                f = Function(V, name=name)

            else:
                raise NotImplementedError

            # Create quadrature function.
            reset_values(f, array.reshape(-1))

        return f

    def mirror(self, X, grad_phi_epi_, grad_phi_lv_, grad_psi_):
        """
        Args:
            X: NumPy array with coordinates of dofs (of current process).
            grads: Fenics Functions.

        Returns:
            Tuple with arrays of mirrored grads.
        """

        # Collect the gradient functions in a list.
        grad_funcs = [grad_phi_epi_, grad_phi_lv_, grad_psi_]

        # Collect the gradient names in a list.
        grad_names = ['grad_phi_epi', 'grad_phi_lv', 'grad_psi']

        # Extract the function space and mesh.
        V_local = grad_phi_epi_.ufl_function_space()
        mesh_local = Mesh(V_local.mesh())

        # Name for temporary HDF5.
        hdf5_filename = 'temporary.hdf5'

        # Open temporary HDF5 file.
        with HDF5File(mpi_comm_world(), hdf5_filename, 'w') as f:
            # Save mesh to temporary HDF5.
            f.write(mesh_local, 'mesh')

            # Save grads to temporary HDF5 file.
            for func, name in zip(grad_funcs, grad_names):
                # Save.
                f.write(func, name)

        # Synchronize.
        MPI.barrier(mpi_comm_world())

        del mesh_local

        # Load entire mesh on local process.
        with HDF5File(mpi_comm_self(), hdf5_filename, 'r') as f:
            mesh = Mesh(mpi_comm_self())
            f.read(mesh, 'mesh', False)

            # Read signature.
            element = f.attributes(grad_names[0])['signature']
            family = element.split(',')[0].split('(')[-1][1:-1]
            degree = int(element.split(',')[2][-2])

        # Create Function space
        V = VectorFunctionSpace(mesh, family, degree)

        # Initialize output tuple.
        out=()

        # For every gradient.
        for idx in range(3):

            # Function of partitioned mesh on local process.
            func_local = grad_funcs[idx]
            name = grad_names[idx]

            # Extract the gradients at the local dofs.
            array_local = func_local.vector().array().reshape(-1, 3)

            # Load function from file entirely on current process.
            func_global = Function(V)
            with HDF5File(mpi_comm_self(), hdf5_filename, 'r') as f:
                f.read(func_global, name)

            # Set extrapolation to True to avoid errors when mirrored point is not inside the domain.
            func_global.set_allow_extrapolation(True)

            # For every dof on local process.
            for ii, x in enumerate(X):

                # For dofs above the equator.
                if x[2] > 0:
                    # Mirror the coords.
                    x_m = x * np.array((1, 1, -1)) # mirror z-coordinate around z = 0.

                    # Calculate gradient of mirrored coordinate.
                    value = func_global(x_m)

                    # Mirror the computed gradient vectors accordingly.
                    if idx == 0:
                        # grad_epi corresponds to transmural vectors.
                        value_m = value * [1, 1, -1]
                    elif idx == 1:
                        # grad_lv corresponds to transmural vectors.
                        value_m = value * [1, 1, -1]
                    elif idx ==2:
                        # grad_psi corresponds to longitudinal vectors.
                        value_m = value * [-1, -1, 1]

                    # Replace the old gradients with the mirrored gradients in ar.
                    array_local[ii, :] = value_m

            # Append the mirrored array to the output variable.
            out += (array_local,)

        # Synchronize.
        MPI.barrier(mpi_comm_world())

        # Delete temporary HDF5 and loaded mesh.
        if MPI.rank(mpi_comm_world()) == 0:
            os.remove(hdf5_filename)
        del mesh

        return out


class CylindricalBasisVectors(object):
    """
    Cylindrical basis vectors defined by z (ez), radial (er), and
    circumferential (ec) values.
    """

    @staticmethod
    def ez(V):
        """
        Cylindrical basis vector oriented along the z-axis.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the coordinate locations of the degrees of freedoms.
        Q = vector_space_to_scalar_space(V)
        X = Q.tabulate_dof_coordinates().reshape(-1, 3)
        # TODO Check that V is in proper space.

        # Prescribe the ec basis vector as [0, 0, 1] for all nodes.
        ez_x = np.zeros(len(X[:, 0]))
        ez_y = np.zeros(len(X[:, 0]))
        ez_z = np.ones(len(X[:, 0]))

        return ez_x, ez_y, ez_z

    @staticmethod
    def er(V):
        """
        Cylindrical basis vector oriented radially around the z-axis.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the coordinate locations of the degrees of freedoms.
        Q = vector_space_to_scalar_space(V)
        X = Q.tabulate_dof_coordinates().reshape(-1, 3)
        # TODO Check that V is in proper space.

        # Split X into x, y, and z components.
        x = X[:, 0]
        y = X[:, 1]
        z = np.zeros(len(X[:, 0]))

        # Compute the radial distance with all three components.
        r = np.sqrt(x**2 + y**2 + z**2)

        # Compute/normalize the radial basis vector.
        er_x = x/r
        er_y = y/r
        er_z = z/r

        return er_x, er_y, er_z

    @staticmethod
    def ec(V, ez, er):
        """
        Cylindrical basis vector oriented circumferentially around the z-axis.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            ez: Cylindrical basis vector aligned along the z-axis.
            er: Cylindrical basis vector aligned radially around the z-axis.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the values of ez and er in array form.
        ez_x, ez_y, ez_z = ez.to_array(V)
        er_x, er_y, er_z = er.to_array(V)
        # TODO Check that V is in proper space.

        # Compute the x, y, and z components of ec.
        ec_x = er_y*ez_z - er_z*ez_y
        ec_y = er_z*ez_x - er_x*ez_z
        ec_z = er_x*ez_y - er_y*ez_x

        # Compute the normalization factor as an array.
        norm = np.sqrt(ec_x**2 + ec_y**2 + ec_z**2)

        # Normalize the relevant components and return.
        ec_x /= norm
        ec_y /= norm
        ec_z /= norm
        return ec_x, ec_y, ec_z


class CylindricalBasisVectorsBiV(object):
    """
    Cylindrical basis vectors defined by z (ez), radial (er), and
    circumferential (ec) values. er is derived from the transmural vector (et)

    # TODO return arrays with nodal values for CoordinateSystem instead of UFL object (like other classes).
    """

    @staticmethod
    def ez():
        """
        Cylindrical basis vector oriented along the z-axis.

        Returns:
            UFL-like object.
        """

        return as_vector([0, 0, 1])

    @staticmethod
    def er(et, ez):
        """
        Cylindrical basis vector oriented radially around the z-axis.

        Derived from wall-bound transmural vector (et) by subtracting the component of et along the ez direction.

        Args:
            et: transmural vector (Function or UFL-like object).
            ez: cylindrical basis vector (Function or UFL-like object).

        Returns:
            UFL-like object.
        """

        er_ = et - inner(et, ez)*ez
        norm = sqrt(er_[0]*er_[0] + er_[1]*er_[1] + er_[2]*er_[2])
        return er_/norm

    @staticmethod
    def ec(ez, er):
        """
        Cylindrical basis vector oriented circumferentially around the z-axis.

        Args:
            mesh (dolfin.Mesh): Mesh to define the vectors on.
            ez: Cylindrical basis vector aligned along the z-axis.
            er: Cylindrical basis vector aligned radially around the z-axis.
        Returns:
            UFL object.
        """
        return -cross(ez, er)


class CylindricalBasisVectorsUFL(object):
    """
    Cylindrical basis vectors defined by z (ez), radial (er), and
    circumferential (ec) values as UFL expressions.
    """

    @staticmethod
    def ez():
        """
        Cylindrical basis vector oriented along the z-axis.

        Returns:
            UFL object.
        """
        # Prescribe the ec basis vector as [0, 0, 1] for all nodes.
        ez = as_vector([0, 0, 1])

        return ez

    @staticmethod
    def er(mesh):
        """
        Cylindrical basis vector oriented radially around the z-axis.

        Args:
            mesh (dolfin.Mesh): Mesh to define the vectors on.

        Returns:
            UFL object.
        """
        # Create symbolic physical coordinates for given mesh.
        X = SpatialCoordinate(mesh)

        # Split X into x and y components.
        x = X[0]
        y = X[1]

        # Compute the radial distance with all three components.
        r = sqrt(x*x + y*y)

        # Compute/normalize the radial basis vector.
        er_x = x/r
        er_y = y/r

        er = as_vector([er_x, er_y, 0])

        return er

    @staticmethod
    def ec(ez, er):
        """
        Cylindrical basis vector oriented circumferentially around the z-axis.

        Args:
            mesh (dolfin.Mesh): Mesh to define the vectors on.
            ez: Cylindrical basis vector aligned along the z-axis.
            er: Cylindrical basis vector aligned radially around the z-axis.
        Returns:
            UFL object.
        """
        # Extract the components of ez and er.
        ez_x, ez_y, ez_z = ez
        er_x, er_y, er_z = er

        # Compute the x, y, and z components of ec.
        ec_x_ = er_y*ez_z - er_z*ez_y
        ec_y_ = er_z*ez_x - er_x*ez_z
        ec_z_ = er_x*ez_y - er_y*ez_x

        # Compute the normalization factor.
        norm = sqrt(ec_x_*ec_x_ + ec_y_*ec_y_ + ec_z_*ec_z_)

        # Normalize the relevant components and return.
        ec_x = ec_x_/norm
        ec_y = ec_y_/norm
        ec_z = ec_z_/norm

        ec = as_vector([ec_x, ec_y, ec_z])
        return ec


class FiberBasisVectors(object):
    """
    Fiber basis vectors defined by fiber (ef), sheet (es), and sheet-normal (en)
    values.

    https://www.ncbi.nlm.nih.gov/pubmed/19592607
    """

    @staticmethod
    def ef(V, ah, at, ec, el, et):
        """
        Fiber basis vector oriented along the fiber direction.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            ah: Helix angle of the fiber field.
            at: Transverse angle of the fiber field.
            ec: Cardiac basis vector oriented in the circumferential axis.
            el: Cardiac basis vector oriented in the longitudinal axis.
            et: Cardiac basis vector oriented in the transmural axis.

        Returns:
            :class:`~numpy.array`
        """
        # TODO Check that V is in proper space.
        Q = vector_space_to_scalar_space(V)

        # Extract the coordinate values of the helix and transverse angles.
        ah_array = ah.to_array(Q)
        at_array = at.to_array(Q)

        # Compute shared component terms.
        cos_at = np.cos(at_array)
        sin_at = np.sin(at_array)
        tan_ah = np.tan(ah_array)
        k1 = np.sqrt(1/(cos_at**2*tan_ah**2 + 1))
        k2 = k1*cos_at*tan_ah

        # Extract the coordinate values of ec, el, and et in array form.
        ec_x, ec_y, ec_z = ec.to_array(V)
        el_x, el_y, el_z = el.to_array(V)
        et_x, et_y, et_z = et.to_array(V)

        # Compute the x, y, and z components of ef.
        ef_x = k1*cos_at*ec_x + k2*el_x + k1*sin_at*et_x
        ef_y = k1*cos_at*ec_y + k2*el_y + k1*sin_at*et_y
        ef_z = k1*cos_at*ec_z + k2*el_z + k1*sin_at*et_z

        # Compute the normalization factor as an array.
        norm = np.sqrt(ef_x**2 + ef_y**2 + ef_z**2)

        # Normalize the relevant components and return.
        ef_x /= norm
        ef_y /= norm
        ef_z /= norm
        return ef_x, ef_y, ef_z

    @staticmethod
    def es(V, at, ec, et):
        """
        Fiber basis vector oriented along the sheet direction.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            at: Transverse angle of the fiber field.
            ec: Cardiac basis vector oriented in the circumferential axis.
            et: Cardiac basis vector oriented in the transmural axis.

        Returns:
            :class:`~numpy.array`
        """
        # TODO Check that V is in proper space.
        Q = vector_space_to_scalar_space(V)

        # Extract the coordinate values of the transverse angles.
        at_array = at.to_array(Q)

        # Compute shared component terms.
        cos_at = np.cos(at_array)
        sin_at = np.sin(at_array)

        # Extract the coordinate values of ec and et in array form.
        ec_x, ec_y, ec_z = ec.to_array(V)
        et_x, et_y, et_z = et.to_array(V)

        # Compute the x, y, and z components of es.
        es_x = sin_at*ec_x - cos_at*et_x
        es_y = sin_at*ec_y - cos_at*et_y
        es_z = sin_at*ec_z - cos_at*et_z

        # Compute the normalization factor as an array.
        norm = np.sqrt(es_x**2 + es_y**2 + es_z**2)

        # Normalize the relevant components and return.
        es_x /= norm
        es_y /= norm
        es_z /= norm
        return es_x, es_y, es_z

    @staticmethod
    def en(V, ef, es):
        """
        Fiber basis vector oriented along the sheet-normal direction.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            ef: Fiber basis vector oriented in the fiber axis.
            es: Fiber basis vector oriented in the sheet axis.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the coordinate values of ef and es in array form.
        ef_x, ef_y, ef_z = ef.to_array(V)
        es_x, es_y, es_z = es.to_array(V)
        # TODO Check that V is in proper space.

        # Compute the x, y, and z components of en.
        en_x = -ef_y*es_z + ef_z*es_y
        en_y = -ef_z*es_x + ef_x*es_z
        en_z = -ef_x*es_y + ef_y*es_x

        # Compute the normalization factor as an array.
        norm = np.sqrt(en_x**2 + en_y**2 + en_z**2)

        # Normalize the relevant components and return.
        #TODO LH or RH system? Division by -norm results in RH.
        en_x /= -norm
        en_y /= -norm
        en_z /= -norm
        return en_x, en_y, en_z


# class RotationAngles(object):
#     """
#     Class to compute and handle rotation angles that represent an fsn basis.
#
#     NOTE: This feature is deprecated.
#           Defining fiber vector basis on QuadratureElement FunctionSpace is preferred
#           over using RotationAngles for the BiV geometry.
#
#     Three rotation angles can be used to rotate a reference basis, i.e. (ex, ey, ez)
#     or (ex, -ey, ez), to get the fiber vector basis.
#     Using these rotation angles, the fiber vectors can be expressed as a UFL expression.
#     Using the representation of fiber vectors as a rotation of a reference basis ensures
#     that the fiber vector basis is always orthonormal at the integration points. This may
#     not be the case if we would use a FEniCS function of the fiber vectors, since then the
#     fiber vectors are independently interpolated at the integration points (not guaranteeing
#     unit length and orthogonality).
#
#     Args:
#         u: The displacement unknown.
#         fiber_vectors: Tuple with basis vectors (ef, es, en) of a orthonormal
#                        coordinate system describing the fiber field.
#                        May be of type array or cvtbk.CoordinateSystem.
#         **kwargs: Arbitrary keyword arguments for user-defined parameters.
#     """
#
#     def __init__(self, fiber_vectors, **kwargs):
#
#         self._parameters = self.default_parameters()
#         self._parameters.update(kwargs)
#
#         self._fiber_vectors = fiber_vectors
#
#         self._rotation_angles_array = None
#
#         self._rotation_angles_sin = None
#         self._rotation_angles_sin_array = None
#
#         self._rotation_angles_cos = None
#         self._rotation_angles_cos_array = None
#
#         self._ef_val, self._es_val, self._en_val = (None, None, None)
#
#         self._Q0 = None
#
#     @property
#     def parameters(self):
#         """
#         Return user-defined parameters for this geometry object.
#         """
#         return self._parameters
#
#     @staticmethod
#     def default_parameters():
#         prm = Parameters('rotation_angles')
#
#         # If True, helix and tranvsverse angles with respect to z-axis are interpolated bidirectionally (ah = ah + k*pi)
#         # However, this works not well (mesh resolution is too low).
#         prm.add('bidirectional_interpolation', False)
#
#     @staticmethod
#     def _create_reference_system(fsn):
#         """
#         Returns a RH or LH reference coordinate system.
#         We'd like to rotate a corresponding reference system (left- or right-handed) to rotate towards fsn.
#         Check whether fsn is a right-handed (RH) or left-handed coordinate (LH) system.
#
#         Args:
#             fsn (3x3 array): An orthonormal basis with the unit vectors as columns.
#
#         Returns:
#             Q0 (list): A corresponding (RH or LH) reference coordinate system.
#         """
#         if np.inner(np.cross(fsn[:,0], fsn[:,1]), fsn[:,2]) > 0.9:
#             # Right-handed system.
#             Q0 = [[1, 0, 0],
#                   [0, 1, 0],
#                   [0, 0, 1]]
#         else:
#             # Left-handed system.
#             Q0 = [[1,  0, 0],
#                   [0, -1, 0],
#                   [0,  0, 1]]
#         return Q0
#
#     def rotation_angles_array(self, V):
#         """
#         Returns a tuple of 3 arrays containing the 3 rotation angles per dof.
#
#         Args:
#             V: the VectorFunctionSpace of the displacement unknown u.
#         """
#         if self._rotation_angles_array is None:
#             try:
#                 fiber_vectors = [_.to_array(V) for _ in self._fiber_vectors]
#             except AttributeError:
#                 fiber_vectors = self._fiber_vectors
#             self._rotation_angles_array = self._create_angles(fiber_vectors)
#
#         return self._rotation_angles_array
#
#     def rotation_angles_cos_array(self, V=None):
#         """
#         Returns a tuple of 3 arrays containing the cosine of the 3 rotation angles per dof.
#
#         Args:
#             V: the VectorFunctionSpace of the displacement unknown u.
#         """
#         if self._rotation_angles_cos_array is None:
#             self._rotation_angles_cos_array = [np.cos(_) for _ in self.rotation_angles_array(V)]
#
#         return self._rotation_angles_cos_array
#
#     def rotation_angles_sin_array(self, V=None):
#         """
#         Returns a tuple of 3 arrays containing the sine of the 3 rotation angles per dof.
#
#         Args:
#             V: the VectorFunctionSpace of the displacement unknown u.
#         """
#         if self._rotation_angles_sin_array is None:
#             self._rotation_angles_sin_array = [np.sin(_) for _ in self.rotation_angles_array(V)]
#
#         return self._rotation_angles_sin_array
#
#     def rotation_angles_cos(self):
#         """
#         Returns a CoordinateSystem object containing the cosine of the 3 rotation angles per dof (as a 3d vector).
#         """
#         if self._rotation_angles_cos is None:
#             self._rotation_angles_cos = CoordinateSystem(self.rotation_angles_cos_array)
#
#         return self._rotation_angles_cos
#
#     def rotation_angles_sin(self):
#         """
#         Returns a CoordinateSystem object containing the sine of the 3 rotation angles per dof (as a 3d vector).
#         """
#         if self._rotation_angles_sin is None:
#             self._rotation_angles_sin = CoordinateSystem(self.rotation_angles_sin_array)
#
#         return self._rotation_angles_sin
#
#     # For validation purposes, we can define ef, es and en as derived from the rotation angles.
#     def ef_val(self, V=None):
#         if self._ef_val is None:
#             self._ef_val, self._es_val, self._en_val = self._create_validation_vectors(V)
#         return self._ef_val
#
#     def es_val(self, V=None):
#         if self._es_val is None:
#             self._ef_val, self._es_val, self._en_val = self._create_validation_vectors(V)
#         return self._es_val
#
#     def en_val(self, V=None):
#         if self._en_val is None:
#             self._ef_val, self._es_val, self._en_val = self._create_validation_vectors(V)
#         return self._en_val
#
#     @staticmethod
#     def rotation_matrix(theta_0, theta_1, theta_2, order=None):
#         """
#         Rotation matrix that rotates with angles theta_i over each of its axis.
#
#         Args:
#             theta_i : angle (in radians) for rotation around axis i.
#             order: tuple/list that specifies the order in which the rotations should be applied.
#                    By default order = (0, 1, 2), i.e. we first rotate with theta_0 around the 0 axis,
#                    then with theta_1 around the (rotated) 1 axis, and finally with theta_2 around the (rotated) 2 axis.
#         """
#
#         rot_0 = np.array([[1, 0, 0],
#                           [0, np.cos(theta_0), -np.sin(theta_0)],
#                           [0, np.sin(theta_0), np.cos(theta_0)]])
#
#         rot_1 = np.array([[np.cos(theta_1), 0, np.sin(theta_1)],
#                           [0, 1, 0],
#                           [-np.sin(theta_1), 0, np.cos(theta_1)]])
#
#         rot_2 = np.array([[np.cos(theta_2), -np.sin(theta_2), 0],
#                           [np.sin(theta_2), np.cos(theta_2), 0],
#                           [0, 0, 1]])
#
#         if order is None:
#             # Default order: First rotate around axis 0, then around axis 1, then around axis 2.
#             return (rot_0 @ rot_1 @ rot_2)
#         else:
#             # Custom order of rotations.
#             rot_all = (rot_0, rot_1, rot_2)
#             return (rot_all[order[0]] @ rot_all[order[1]] @ rot_all[order[2]])
#
#     @staticmethod
#     def find_rotations(fsn, right_handed):
#         """
#         Finds the rotation angles to rotate a reference orthonormal coordinate system
#         towards an arbitrary coordinate system (fsn).
#         The angles represent rotations around each axis: i.e. theta_0 is a rotation around
#         axis 0, theta_1 is a rotation around axis 1, etc.
#
#         Note that the order of applying the rotations is of importance.
#         This code computes the rotation angles (theta_0, theta_1, theta_2) that rotate
#         a reference coordinate system by first rotating around axis 2, then axis 1,
#         and finally axis 0, with the corresponding angles as explained above.
#
#         The code solves
#
#         Q * R_2(theta_2) * R_1(theta_1) * R_0(theta_0) = fsn
#
#         analytically for theta_0, theta_1, theta_2,
#         where Q is a right-handed (RH) or left-handed (LH) reference coordinate system:
#
#         RH: [[1, 0, 0],
#              [0, 1, 0],
#              [0, 0, 1]]
#
#         LH: [[1,  0, 0],
#              [0, -1, 0],
#              [0,  0, 1]]
#
#          and where R_i is a rotation matrix that rotates with a rotation angle theta_i around axis i.
#
#         Args:
#              fsn (3x3 array): An orthonormal basis with the unit vectors as columns.
#              right_handed (bool): Specify whether fsn is a right- or left-handed coordinate system.
#
#         Returns:
#             Tuple (theta_0, theta_1, theta_2) with angles around which to rotate a Cartesian basis to get the fsn basis.
#             Note that the order of the rotations is important: we first need to rotate around axis 2, then 1 and finally 0.
#             I.e.: fsn = rotation_matrix(theta_0, theta_1, theta_2, order=[2,1,0])
#         """
#         theta_0 = np.arctan2(fsn[2, 1], fsn[2, 2])
#         if right_handed:
#             theta_2 = np.arctan2(fsn[1, 0], fsn[0, 0])
#         else:
#             # left-handed system.
#             theta_2 = np.arctan2(-fsn[1, 0], fsn[0, 0])
#
#         if abs(fsn[0, 0]) > 1e-10:
#             theta_1 = np.arctan2(-fsn[2, 0], fsn[0, 0] / np.cos(theta_2))
#         elif abs(fsn[1, 0]) > 1e-10:
#             theta_1 = np.arctan2(-fsn[2, 0], fsn[1, 0] / np.sin(theta_2))
#         else:
#             # Implies that cos(theta_1) = 0, determine theta_1 just from sine.
#             theta_1 = np.arcsin(fsn[2, 0])
#
#         return theta_0, theta_1, theta_2
#
#     @staticmethod
#     def find_ahz_atz(ef):
#         """
#         Find helix and transverse angles of ef with respect to z-axis, biderectionally (on domain [0, pi]).
#
#         Args:
#             ef: array-like with fiber vector.
#
#         Returns:
#             ahz, atz: helix and tranverse angles, on domain [0, pi].
#
#         """
#         ahz = np.arctan2(np.inner(ef, [1, 0, 0]), np.inner(ef, [0, 0, 1]))
#         atz = np.arctan2(np.inner(ef, [0, 1, 0]), np.inner(ef, [0, 0, 1]))
#
#         # Direction of fiber vector is arbitrary: ah = ah + k*pi.
#         # -> Define all angles on domain [0, pi].
#         if ahz<0:
#             ahz += np.pi
#         if atz<0:
#             atz += np.pi
#         return ahz, atz
#
#     def _create_angles(self, fiber_vectors):
#         """
#         Computes the rotation angles for all fsn basises in fiber_vectors.
#
#         Args:
#             fiber_vectors: Tuple of arrays (ef, es, en) containing unit vectors at dofs
#                            (e.g. fiber_vectors[0].shape = (ndofs, 3))
#
#         Returns:
#             Tuple of arrays (theta_0, theta_1, theta_2) with the rotation angles at all nodes
#             that can transform a reference basis to an fsn basis (see self.find_rotations).
#         """
#         bidir = self.parameters['bidirectional_interpolation']
#
#         # Convert tuples to array.
#         fsn_all = np.asarray(fiber_vectors)  # shape = (3, 3, ndfos)
#
#         # Initialize the arrays for theta's.
#         ndofs = fsn_all.shape[2]
#
#         if bidir:
#             ahz = np.zeros(ndofs)
#             atz = np.zeros(ndofs)
#             dummy = np.zeros(ndofs)
#         else:
#             theta_0 = np.zeros(ndofs)
#             theta_1 = np.zeros(ndofs)
#             theta_2 = np.zeros(ndofs)
#
#         # Loop over dofs.
#         for ii in range(ndofs):
#             if bidir:
#                 # Find helix and tranverse angles with respect to z-axis.
#                 ef = [fiber_vectors[0][0][ii], fiber_vectors[0][1][ii], fiber_vectors[0][2][ii]]
#                 ahz[ii], atz[ii] = self.find_ahz_atz(ef)
#
#             else:
#                 # Extract fsn orthogonal matrix. We need the unit vectors as columns, so we have to transpose.
#                 fsn = np.transpose(fsn_all[:, :, ii])
#
#                 if ii == 0:
#                     # Check handedness of fsn to determine reference system.
#                     self._Q0 = self._create_reference_system(fsn)
#                     right_handed = True if np.sum(self._Q0) == 3 else False
#
#                 # Find rotation angles.
#                 theta_0[ii], theta_1[ii], theta_2[ii] = self.find_rotations(fsn, right_handed)
#
#         if bidir:
#             # Multiply the angles by 2, to perform bidirectional interpolation using the cosines and sines.
#             return 2*ahz, 2*atz, dummy
#
#         else:
#             # Validate that the rotated Cartesian basis with the found angles leads to fsn (for one case to preserve speed).
#             R_xyz_to_fsn = np.array(self._Q0) @ self.rotation_matrix(theta_0[-1], theta_1[-1], theta_2[-1],
#                                                                      order=[2, 1, 0])
#             assert np.sum(np.abs(
#                 fsn - R_xyz_to_fsn)) < 1e-8, 'fsn derived from rotation angles is not the same as initial fsn, error: {}.'.format(
#                 np.sum(np.abs(fsn - R_xyz_to_fsn)))
#
#             return theta_0, theta_1, theta_2
#
#
#     def _create_validation_vectors(self, V=None):
#         """
#         Helper function that computes the fiber vector basis from the derived rotation angles at all dofs.
#         """
#         bidir = self.parameters['bidirectional_interpolation']
#
#         if bidir:
#             sin_ahz, sin_atz = [self.rotation_angles_sin().to_array(V)[i] for i in range(2)]
#             cos_ahz, cos_atz = [self.rotation_angles_cos().to_array(V)[i] for i in range(2)]
#
#             ahz = 0.5*np.arctan2(sin_ahz, cos_ahz)
#             atz = 0.5*np.arctan2(sin_atz, cos_atz)
#
#             # Reference coordinate system for orient function in BasisVectorsBayer (ez, ex, ey).
#             Q0 = ([0, 0, 1], [1, 0, 0], [0, 1, 0])
#
#             ndofs = len(sin_ahz)
#         else:
#             sin_theta_0, sin_theta_1, sin_theta_2 = [self.rotation_angles_sin().to_array(V)[i] for i in range(3)]
#             cos_theta_0, cos_theta_1, cos_theta_2 = [self.rotation_angles_cos().to_array(V)[i] for i in range(3)]
#
#             theta_0 = np.arctan2(sin_theta_0, cos_theta_0)
#             theta_1 = np.arctan2(sin_theta_1, cos_theta_1)
#             theta_2 = np.arctan2(sin_theta_2, cos_theta_2)
#
#             ndofs = len(theta_0)
#
#         ef_val = np.zeros((ndofs, 3))
#         es_val = np.zeros((ndofs, 3))
#         en_val = np.zeros((ndofs, 3))
#
#
#         for ii in range(ndofs):
#             if bidir:
#                 fsn = BasisVectorsBayer.orient(Q0, ahz[ii], atz[ii])
#                 ef_val[ii, :] = fsn[0]
#                 es_val[ii, :] = fsn[1]
#                 en_val[ii, :] = fsn[2]
#
#             else:
#                 fsn = np.array(self._Q0) @ self.rotation_matrix(theta_0[ii], theta_1[ii], theta_2[ii], order=[2, 1, 0])
#                 ef_val[ii, :] = fsn[:, 0]
#                 es_val[ii, :] = fsn[:, 1]
#                 en_val[ii, :] = fsn[:, 2]
#
#         ef_val = [ef_val[:, idx] for idx in range(3)]
#         es_val = [es_val[:, idx] for idx in range(3)]
#         en_val = [en_val[:, idx] for idx in range(3)]
#
#         return ef_val, es_val, en_val
#
#     @staticmethod
#     def orient_ahz_atz_ufl(ahz, atz):
#         """
#         UFL version of BasisVectorsBayer.orient(), with fixed reference coordinate system (ez, ex, ey).
#         """
#         Q = as_matrix([[0, 1, 0],
#                        [0, 0, 1],
#                        [1, 0, 0]])
#
#         # Calculate gamma (transverse angle in ec'-et plane (after rotation with helix angle ah)).
#         f_c = 1/sqrt((tan(ahz))**2 + (tan(atz))**2 + 1)
#         f_l = tan(ahz) * f_c
#         f_t = tan(atz) * f_c
#         gamma = atan2_(f_t, sqrt(f_c**2 + f_l**2))
#
#         # Create rotation matrices.
#         # First rotation: rotate around et with angle ah.
#         R_ah = as_matrix([[cos(ahz), -sin(ahz), 0],
#                         [sin(ahz), cos(ahz), 0],
#                         [0, 0, 1]])
#
#         # Second rotation: rotate around el' with angle -gamma (the tranverse angle in ec'-el plane).
#         R_at = as_matrix([[cos(-gamma), 0, sin(-gamma)],
#                         [0, 1, 0],
#                         [-sin(-gamma), 0, cos(-gamma)]])
#
#         # Rotate Q.
#         fsn = Q * R_ah * R_at
#
#         return fsn
#
#     def rotation_angles_to_fsn(self, V):
#         """
#         Return UFL expression that expresses the fsn basis as a function of rotation angles (ef, es, en).
#
#         Args:
#             V: VectorFunctionSpace to define the fiber vectors on.
#
#         Returns:
#             Tuple with UFL expressions (ufl.tensors.ComponentTensor) for the fsn basis (ef, es, en).
#         """
#         bidir = self.parameters['bidirectional_interpolation']
#
#         if bidir:
#             sin_ahz, sin_atz = [self.rotation_angles_sin().to_function(V)[i] for i in range(2)]
#             cos_ahz, cos_atz = [self.rotation_angles_cos().to_function(V)[i] for i in range(2)]
#
#             # Reconstruct helix and transverse angles bidirectionally.
#             #TODO use UFL atan2 if fixed.
#             ahz = 0.5*atan2_(sin_ahz, cos_ahz)
#             atz = 0.5*atan2_(sin_atz, cos_atz)
#
#             fsn = self.orient_ahz_atz_ufl(ahz, atz)
#
#         else:
#             sin_theta_0, sin_theta_1, sin_theta_2 = [self.rotation_angles_sin().to_function(V)[i] for i in range(3)]
#             cos_theta_0, cos_theta_1, cos_theta_2 = [self.rotation_angles_cos().to_function(V)[i] for i in range(3)]
#
#             #TODO use UFL atan2 if fixed.
#             theta_0 = atan2_(sin_theta_0, cos_theta_0)
#             theta_1 = atan2_(sin_theta_1, cos_theta_1)
#             theta_2 = atan2_(sin_theta_2, cos_theta_2)
#
#             # We first rotate around the last axis with theta_2.
#             R_1 = as_matrix([[cos(theta_2), -sin(theta_2), 0],
#                              [sin(theta_2), cos(theta_2), 0],
#                              [0, 0, 1]])
#
#             # Next, we rotate around the middle axis with theta_1.
#             R_2 = as_matrix([[cos(theta_1), 0, sin(theta_1)],
#                              [0, 1, 0],
#                              [-sin(theta_1), 0, cos(theta_1)]])
#
#             # Ultimately, we rotate around the first axis with theta_0.
#             R_3 = as_matrix([[1, 0, 0],
#                              [0, cos(theta_0), -sin(theta_0)],
#                              [0, sin(theta_0), cos(theta_0)]])
#
#             Q0 = as_matrix(self._Q0)
#             fsn = Q0 * R_1 * R_2 * R_3
#
#         ef, es, en = [fsn[:, ii] for ii in range(3)]
#
#         return ef, es, en
#
#     def print_error(self, figpath=None, V=None):
#         """
#         Prints the error of the reconstructed ef, derived from the rotation angles.
#         If figpath is given, a histogram of the error is plotted and saved to figpath.
#         """
#
#         # Error with validation vectors at nodes.
#         # Compute reconstructed fiber vectors from rotation angles.
#         ef_val = self.ef_val(V)
#
#         # Extract original fiber vectors.
#         ef = self._fiber_vectors[0].to_array(V)
#
#         # Compute error.
#         error = []
#         for i in range(len(ef[0])):
#             ef_i = np.array([ef[0][i], ef[1][i], ef[2][i]])
#             ef_p_i = np.array([ef_val[0][i], ef_val[1][i], ef_val[2][i]])
#             inner = np.max((np.inner(-1 * ef_i, ef_p_i), np.inner(ef_i, ef_p_i)))
#             error.append(np.arccos(np.clip(inner, -1, 1))*180/np.pi)
#         error = np.array(error)
#
#         print_once('Sum error (degrees):', np.sum(error))
#         print_once('Mean error (degrees):', np.mean(error))
#         if figpath is not None:
#             import matplotlib.pyplot as plt
#             plt.hist(error); plt.title('Abs difference between components of original and reconstructed ef')
#             plt.savefig(figpath)


class GeoFunc(object):
    """
    Class containing static helper functions that are used for correcting the transmural (v) and longitudinal (u)
    coordinates as derived from solutions of the Laplace equation.

    NOTE: This is probably not the best place for this class, but putting it in geometries.py does not work either, as
    that module imports this one.
    """
    @staticmethod
    def ellipsoidal_to_cartesian(sig, tau, phi, focus):
        X = np.zeros((len(sig), 3))

        term = np.sqrt((sig ** 2 - 1) * (1 - tau ** 2))
        X[:, 0] = focus * term * np.cos(phi)
        X[:, 1] = focus * term * np.sin(phi)
        X[:, 2] = focus * sig * tau

        return X

    @staticmethod
    def analytical_u(X, sig, tau, cutoff, linearlize=True):
        """
        Wall-bounded normalized longitudinal position defined in [-1, 0.5] (linearize=True).

        Args:
            X: array of shape (n, 3): contains the x,y,z for n points to define the coordinate on.
            sig: Ellipsoidal radial position σ.
            tau: Ellipsoidal longitudinal position τ.
            cutoff: Distance above the xy-plane to truncate the ellipsoids.
            linearlize (boolean): Linearizes the longitudinal coordinate with respect to z.
        Returns:
            :class:`~numpy.array`
        """
        # Extract z-coordinates.
        z = X[:, 2]

        # Create an empty array for future v values.
        u_array = np.zeros(len(sig))

        # Normalize values of u above origin plane in a linear manner.
        if linearlize:
            u_array[z > 0] = 0.5 / cutoff * z[z > 0]

        # Define a function for SciPy's integrator to return wall distances.
        def fx(x_val, s_val):
            # Formula is scale factors from
            # https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates
            return np.sqrt(s_val ** 2 - x_val ** 2) / np.sqrt(1 - x_val ** 2)

        # Integrate for each point to obtain the normalized v distances.
        for i, s in enumerate(sig):
            if z[i] <= 0 or linearlize==False:
                AB = scipy.integrate.quad(fx, 0, -1, args=(s,))
                AP = scipy.integrate.quad(fx, 0, tau[i], args=(s,))
                u_array[i] = -AP[0] / AB[0]

        # Return the normalized u coordinate values.
        return u_array

    @staticmethod
    def compute_phi_sep_endo(geo_pars, z=0):
        """
        Computes the angle (phi) at which the epicardial septal LV wall attaches to the RV endocardium in the xy-plane.
        Args:
            geo_pars (dolfin.Parameters): geometric parameters for BiventricleGeometry.
            z: z-coordinate.
        Returns:
            The angle phi in radians.
        """
        # Septum radii
        a_1 = geo_pars['R_2sep']
        b_1 = geo_pars['R_2']
        c_1 = geo_pars['Z_2']

        # RV endocardium radii
        a_2 = geo_pars['R_3']
        b_2 = geo_pars['R_3y']
        c_2 = geo_pars['Z_3']

        # Compute y-coordinate where the seotal LV epicardium and RV endocardium meet.
        nom = 1 - (a_1 / a_2) ** 2 * (1 - (z / c_1) ** 2) - (z / c_2) ** 2
        den = (1 / b_2) ** 2 - (a_1 / a_2 / b_1) ** 2
        y = np.sqrt(nom / den)

        # Compute corresponding x-coordinate (the negative one, RV is to the left of LV).
        x = -a_1 * np.sqrt(1 - (y / b_1) ** 2 - (z / c_1) ** 2)

        # Return corresponding angle.
        return np.arctan2(a_1 * y, b_1 * x)

    @staticmethod
    def compute_sigma(x, y, z, focus):
        """
        Ellipsoidal radial position defined such that constant values represent
        an ellipsoidal surface.

        Args:
            x, y, z: x, y, z coordinates.
            focus: focal length of ellipsoid.
        Returns:
            sigma
        """
        # Compute and return the sigma values.
        return 0.5 * (np.sqrt(x ** 2 + y ** 2 + (z + focus) ** 2)
                      + np.sqrt(x ** 2 + y ** 2 + (z - focus) ** 2)) / focus

    @staticmethod
    def compute_tau(x, y, z, focus):
        """
        Ellipsoidal longitudinal position defined in [-1, 1].

        Args:
            x, y, z: x, y, z coordinates.
            focus: focal length of ellipsoid.
        Returns:
            tau
        """
        # Compute and return the tau values.
        return 0.5 * (np.sqrt(x ** 2 + y ** 2 + (z + focus) ** 2)
                      - np.sqrt(x ** 2 + y ** 2 + (z - focus) ** 2)) / focus

    @staticmethod
    def compute_x_sep(y, z, R_2, R_2sep, Z_2):
        """Compute (negative) x-coordinate of septal ellipsoid at y=y and z=z.
        # (x/a)**2 = (1 - (y/b)**2 - (z/c)**2) with a = R_2sep, b = R_2 and c = Z_2
        Args:
            y, z: y and z coordinate.
            R_2, R_2sep, Z_2: geometric parameters
        Returns:
            left-hand (negative) x-coordinate corresponding to point on ellips with given y and z coordinate.
        """
        # y and z should not exceed the principal axis of the ellipsoid.
        y_clipped = np.clip(y, -R_2, R_2)
        z_clipped = np.clip(z, -Z_2, Z_2)

        # Compute (negative) x-coordinate of septal ellipsoid.
        x_septum = -np.sqrt(R_2sep ** 2 * (1 - (y_clipped / R_2) ** 2 - (z_clipped / Z_2) ** 2))

        return x_septum

    @staticmethod
    def calculate_coords(a, b, c, theta, phi):
        """
        Returns x, y, z coordinate of ellipsoid as given by:
        x = a cos(theta) cos(phi)
        y = b cos(theta) sin(phi)
        z = c sin(theta)

        Returns:
            numpy array with x, y, z coordinate.
        """
        x = a * np.cos(theta) * np.cos(phi)
        y = b * np.cos(theta) * np.sin(phi)
        z = c * np.sin(theta)
        return np.asarray((x, y, z))

    @staticmethod
    def find_closest_distance(x, y, z, a, b, c, d_phi=0.01, d_theta=0.01):
        """
        Finds the closest distance of point x,y,z to ellipsoid defined by principal semi-axis a, b, c.
        """
        # Create an initial guess for phi and theta.
        phi0 = np.arctan2(a * y, b * x)
        theta0 = np.arcsin(np.clip(z / c, -1, 1))
        # Collect target coordinate in array.
        coord = np.asarray((x, y, z))
        # Find phi
        # Initial distance to x, y, z.
        coord1 = GeoFunc.calculate_coords(a, b, c, theta0, phi0)
        d1 = norm(coord - coord1)
        ddist = -1
        first_iter = True
        while ddist < 0:
            d0 = d1 * 1
            # Make a step in phi
            phi0 += d_phi
            coord1 = GeoFunc.calculate_coords(a, b, c, theta0, phi0)
            d1 = norm(coord - coord1)
            ddist = d1 - d0
            if first_iter:
                if ddist > 0:
                    # Go other way round.
                    d_phi = -1 * d_phi
                    ddist = -1
                first_iter = False

        # Find theta.
        ddist = -1
        first_iter = True
        while ddist < 0:
            d0 = d1 * 1
            # Make a step in theta.
            theta0 += d_theta
            coord1 = GeoFunc.calculate_coords(a, b, c, theta0, phi0)
            d1 = norm(coord - coord1)
            ddist = d1 - d0
            if first_iter:
                if ddist > 0:
                    # Go other way round.
                    d_theta = -1 * d_theta
                    ddist = -1
                first_iter = False
        # Return previous distance (which is the minimum distance).
        return d0

    @staticmethod
    def analytical_v(sig, tau, sigma_inner, sigma_outer):
        """
        Analytical wall-bounded normalized radial position defined in [0, 1] such that
        constant values represent an ellipsoidal surface.

        Args:
            sig: Ellipsoidal radial position σ.
            tau: Ellipsoidal longitudinal position τ.
            sigma_inner: Value of σ on the inner surface.
            sigma_outer: Value of σ on the outer surface.

        Returns:
            v
        """
        # Define a function for SciPy's integrator to return wall distances.
        def fx(x_val, t_val):
            # Formula is scale factors from
            # https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates
            return np.sqrt(x_val ** 2 - t_val ** 2) / np.sqrt(x_val ** 2 - 1)

        # Integrate to obtain the normalized v distances.
        AB = scipy.integrate.quad(fx, sigma_inner, sigma_outer, args=(tau,))
        AP = scipy.integrate.quad(fx, sigma_inner, sig, args=(tau,))

        # Return the normalized v coordinate [0, 1].
        return AP[0] / AB[0]

    @staticmethod
    def closest_distance_v(x, y, z, a_endo, a_epi, b_endo, b_epi, c_endo, c_epi, d_phi=0.01, d_theta=0.01):
        """
        Computes the normalized coordinate v, by finding the closest distance to the endocardium and epicardum ellipsoid.
        It finds corresponding theta and phi on endocardium and epicardium in parameterized description of the ellipsoid
        by minimizing the distance with given x, y, z:

        for i = epi, endo:
        xp = a_i cos(theta) cos(phi)
        yp = b_i cos(theta) sin(phi)
        zp = c_i sin(theta)
        find (theta, phi) that minimizes ((x-xp)**2 + (y-yp)**2 + (z-zp)**2)
        """
        # Find distance to endocard.
        d_endo = GeoFunc.find_closest_distance(x, y, z, a_endo, b_endo, c_endo, d_phi=d_phi, d_theta=d_theta)
        # Find distance to epicard.
        d_epi = GeoFunc.find_closest_distance(x, y, z, a_epi, b_epi, c_epi, d_phi=d_phi, d_theta=d_theta)
        # Compute normalized transmural coordinate.
        v = d_endo / (d_endo + d_epi)

        # Return the normalized v coordinate [0, 1].
        return v

    @staticmethod
    def follow_et_v(x, y, z, et, d_x, mesh):
        """
        Another (not preferred) method that tries to define the wall bound coordinate.
        Is not based on analytical description, but follows the transmural unit vectors untill the end of the mesh (endo, epi).
        Not a preferred method, since the unit vectors may not be accurate enough. Another problem arises at the base, even when
        unit vectors are accurate: the transmural direction is not parallel to the base, meaning that the end of the mesh is
        reached before arriving at the epicardium.
        Another concern is that I do not know if this will work when the mesh is partitioned when running parallelized.
        However, when using patient specific meshes (without analytical description), this might be the start of a method to define WB coordinates.
        """
        # Initial point:
        xi = (x, y, z)
        # Find epicardium.
        x0 = np.copy(xi)
        while mesh.bounding_box_tree().collides_entity(Point(x0)):
            x0 += et * d_x
        # Half a step back to estimate epicardium border.
        x_epi = x0 - et * d_x / 2

        # Find endocardium.
        x0 = np.copy(xi)
        while mesh.bounding_box_tree().collides_entity(Point(x0)):
            x0 -= et * d_x
        # Half a step forward to estimate endocardium border.
        x_endo = x0 + et * d_x / 2

        AP = norm(x_endo - xi)
        AB = norm(x_endo - x_epi)
        return AP / AB

    @staticmethod
    def interpolate_v_septum(geometry, v_septum_func):
        """
        Determines the relation of Bayers transmural coordinate in the septum with the true normalized linear
        transmural coordinate at y=z=0 and x = [-R_1sep, -R_2sep].
        Then, interpolates the found relation to all other points to approximate the linear transmural coordinate.

        Args:
            geometry: a cvbtk BiventricularGeometry with 'geomertry' parameters and a mesh.
            v_septum_func: a FEniCS function that contains the septal normalized transmural coordinate as used by Bayer et al. 2012
        Returns:
            A FEniCS function of the corrected septal transmural normalized coordinate.
        """
        # Find relation between normalized transmural coordinate d and travelled distance at
        # septum at y=z=0
        r_inner = -geometry.parameters['geometry']['R_1sep']
        r_outer = -geometry.parameters['geometry']['R_2sep']
        d_wall = abs(r_outer - r_inner) / 50  # Prevent probing outside the mesh/domain.
        x_coords = np.linspace(r_inner + d_wall, r_outer - d_wall, 200)
        v_transmural = np.zeros(len(x_coords))
        true_distance = np.zeros(len(x_coords))
        for ii, x in enumerate(x_coords):
            true_distance[ii] = abs(x - x_coords[0])
            try:
                v_transmural[ii] = v_septum_func(x, 0, 0)
            except:
                v_transmural[ii] = -1e16

            # Gather the evaluations (some processes may not contain the point).
            v_transmural[ii] = MPI.max(mpi_comm_world(), v_transmural[ii])

        true_distance -= true_distance[0]
        true_distance = abs(true_distance)
        # Convert to [0,1] domain
        true_distance /= max(true_distance)

        idx_include = v_transmural > -1
        v_transmural = v_transmural[idx_include]
        true_distance = true_distance[idx_include]

        # Interpolate.
        true_dist_sep = scipy.interpolate.interp1d(v_transmural, true_distance, kind='quadratic',
                                             fill_value=(true_distance[0], true_distance[-1]), bounds_error=False)

        # Calculate interpolated nodal transmural coordinates.
        v_septum_i = true_dist_sep(v_septum_func.vector().array())

        # Convert to FEniCS function and return.
        v_septum_func_i = Function(v_septum_func.ufl_function_space())
        v_septum_func_i.vector()[:] = v_septum_i
        v_septum_func_i.vector().apply('')
        return v_septum_func_i
