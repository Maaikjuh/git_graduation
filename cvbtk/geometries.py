# -*- coding: utf-8 -*-
"""
This module provides high-level geometry classes that have interfaces to meshes,
relevant coordinate systems and basis vectors, and other useful methods such as
volume computation routines.
"""
import warnings

import numpy as np
import scipy.optimize
import functools
import os

from dolfin import Function, VectorFunctionSpace, interpolate, unit_vector, project, \
    FunctionSpace, VectorElement, parameters
from dolfin.common.constants import DOLFIN_PI
from dolfin.cpp.common import MPI, Parameters, Timer, mpi_comm_world, mpi_comm_self
from dolfin.cpp.function import ALE
from dolfin.cpp.io import HDF5File, XDMFFile
from dolfin.cpp.mesh import (BoundaryMesh, Facet, FacetFunction, Mesh, cells,
                             facets, vertices, Cell, CellFunction)

from ufl import Measure

from .utils import vector_space_to_scalar_space, save_to_disk, print_once, reset_values, quadrature_function_space
from .basis_vectors import (CardiacBasisVectors, CylindricalBasisVectors,
                            FiberBasisVectors, BasisVectorsBayer, CylindricalBasisVectorsBiV)
from .coordinate_systems import (CoordinateSystem, EllipsoidalCoordinates,
                                 WallBoundedCoordinates)
from .fiber_angles import DefaultFiberAngles, ComputeFiberAngles
from .meshes import (LeftVentricleMesh, LeftVentricleVADMesh,
                     BiventricleMesh, compute_biventricular_parameters,
                     ThickWalledSphereMesh)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = [
    'BaseGeometry',
    'LeftVentricleGeometry',
    'BiventricleGeometry',
    'LeftVentricleVADGeometry',
    'ThickWalledSphereGeometry'
]


class BaseGeometry(object):
    """
    Geometry-independent methods are defined here to facilitate re-use in new
    geometry class definitions.

    Args:
        meshfile (optional): Path to an existing mesh to load in.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, meshfile=None, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        self._tags = None
        self._fiber_vectors = None

        # Load the mesh from meshfile if given.
        if meshfile:
            file_extension = os.path.splitext(meshfile)[1]
            if file_extension == '.hdf5' or file_extension == '.h5':
                with HDF5File(mpi_comm_world(), meshfile, 'r') as f:
                    self._mesh = Mesh()
                    f.read(self._mesh, 'mesh', False)
            elif file_extension == '.xml':
                self._mesh = Mesh(meshfile)
            else:
                raise NotImplementedError('File {} not supported.'.format(meshfile))
            # TODO Add reading from XDMF files, if possible.
        else:
            self._mesh = None

    @staticmethod
    def default_parameters():
        raise NotImplementedError

    @property
    def parameters(self):
        """
        Return user-defined parameters for this geometry object.
        """
        return self._parameters

    def compute_volume(self, *args, **kwargs):
        return NotImplementedError

    def fiber_vectors(self):
        """
        Return the associated fiber basis vectors for this geometry.
        """
        if not self._fiber_vectors:
            self._fiber_vectors = self._create_fiber_vectors()
        return self._fiber_vectors

    def load_fiber_field(self, filepath=None, openfile=None, vector_number=0):
        """
        Loads the saved fiber vectors ef to the geometry.

        Args:
            Either specify the path to the file containing te fiber_vector (filepath) or pass an open file (openfile).
            vector_number (optional): Specify the vector number to load (if multiple fiber_vector datasets are stored).
        """
        if openfile is None:
            openfile = HDF5File(mpi_comm_world(), filepath, 'r')

        if openfile.has_dataset('fiber_vector'):
            # Retrieve element signature
            attr = openfile.attributes('fiber_vector')
            element = attr['signature']
            family = element.split(',')[0].split('(')[-1][1:-1]
            cell = element.split(',')[1].strip()
            degree = int(element.split(',')[2])
            quad_scheme = element.split(',')[3].split('=')[1].split(')')[0][1:-1]
            # Check if the loaded quadrature degree corresponds to te current quadrature degree.
            if degree != parameters['form_compiler']['quadrature_degree']:
                warnings.warn(
                    ("\nThe quadrature degree of the loaded fiber vectors (= {}) is not the same as \nthe current " +
                    "quadrature degree in parameters['form_compiler']['quadrature_degree']").format(degree))
            # Create function space.
            element_V = VectorElement(family=family,
                                      cell=cell,
                                      degree=degree,
                                      quad_scheme=quad_scheme)
            V = FunctionSpace(self.mesh(), element_V)
            ef_func = Function(V)
            try:
                # Maybe multiple fiber vectors are saved, load the correct vector.
                ef_vector = '{}/vector_{}'.format('fiber_vector', vector_number)
                openfile.read(ef_func, ef_vector)
            except RuntimeError:
                ef_vector = 'fiber_vector'
                openfile.read(ef_func, ef_vector)
            print_once('Loading fiber vectors from file, dataset {}'.format(ef_vector))
        else:
            print_once('No fiber field saved.')
            return

        # Collect fiber vectors in array.
        ef_array = ef_func.vector().array().reshape((-1,3))
        # Compute es and en.
        es_array = np.zeros(ef_array.shape)
        en_array = np.zeros(ef_array.shape)
        for i, ef in enumerate(ef_array):
            # Compute sheet direction (some vector orthogonal to ef, exact direction does not matter).
            es = np.array([-ef[1], ef[0], 0])
            # Normalize.
            es_array[i, :] = es / np.linalg.norm(es)
            # Compute sheet-normal direction (orthogonal to ef and es).
            en = np.cross(ef, es)
            # Normalize.
            en_array[i, :] = en / np.linalg.norm(en)
        # Reset the reference fiber vectors.
        self.set_fiber_vectors(ef_array, es_array, en_array, V)
        return

    def mesh(self):
        """
        Return the associated :class:`~dolfin.Mesh` for this geometry.
        """
        if not self._mesh:
            self._mesh = self._create_mesh()
        return self._mesh

    def set_fiber_vectors(self, ef, es, en, V=None):
        """
        Sets the fiber vectors with new ef, es and en.

        Args:
            ef, es, en : arrays with the fiber, sheet, and sheet-normal vectors, respectively, for all nodes.
        """
        # Convert to (-1, 3) arrays (ordering of arrays is important for resetting the values of FEniCS functions.
        if type(ef) is tuple:
            ef = np.transpose(np.vstack((ef[0], ef[1], ef[2])))
            es = np.transpose(np.vstack((es[0], es[1], es[2])))
            en = np.transpose(np.vstack((en[0], en[1], en[2])))
        else:
            # If not tuple, assume that ef, es and en are numpy arrays.
            if ef.shape[0] == 3:
                # Convert (3, -1) arrays to (-1, 3) arrays.
                ef = np.transpose(np.vstack((ef[0, :], ef[1, :], ef[2, :])))
                es = np.transpose(np.vstack((es[0, :], es[1, :], es[2, :])))
                en = np.transpose(np.vstack((en[0, :], en[1, :], en[2, :])))
            elif ef.shape[1] == 3:
                # Shape is good.
                pass
            else:
                raise ValueError('Unexpected shape of fiber vectors. Shape ef is {}.'.format(ef.shape))

        # Reset fiber_vectors.
        e_new_tuples = (ef, es, en)
        for i in range(3):
            # Create array with new nodal values with correct ordering of the dofs.
            e_new_array = e_new_tuples[i].reshape(-1)

            # Reset the arrays of the fiber_vectors objects.
            self.fiber_vectors()[i]._array = (e_new_tuples[i][:, 0], e_new_tuples[i][:, 1], e_new_tuples[i][:, 2])

            # Get the FEniCS function that holds the vectors.
            if self.fiber_vectors()[i]._function is not None:
                # Vector function is already created, we do not need to specify the function space.
                e_f = self.fiber_vectors()[i].to_function(None)

                # Assign the new values to the FEniCS function.
                reset_values(e_f, e_new_array)

            elif V is not None:
                # Function is not yet creates. If a vector FunctionSpace is provided, create function.
                self.fiber_vectors()[i].to_function(V)

            else:
                # The function object of the CoordinateSystem will be created later,
                # when called with a suitable FunctionSpace. Having set the _array property is enough for now.
                pass

    def tags(self):
        """
        Return the associated :class:`~dolfin.MeshFunction` for this geometry.
        """
        if not self._tags:
            self._tags = self._create_tags()
        return self._tags

    def _create_fiber_vectors(self):
        """
        Helper method to create the fiber (ef, es, en) basis vectors.
        """
        ef, es, en = [unit_vector(i, 3) for i in range(3)]
        return ef, es, en

    def _create_mesh(self):
        raise NotImplementedError

    def _create_tags(self):
        raise NotImplementedError

    def cardiac_vectors(self):
        raise NotImplementedError

    def wallbounded_coordinates(self):
        raise NotImplementedError

    def fiber_angles(self):
        raise NotImplementedError

class LeftVentricleGeometry(BaseGeometry):
    """
    High-level interface to a left ventricle for simulation and post-processing.

    Args:
        meshfile (optional): Path to an existing mesh to load in.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, meshfile=None, **kwargs):
        super(LeftVentricleGeometry, self).__init__(meshfile=meshfile, **kwargs)

        # Define this here, before reloading mesh and fiber vectors.
        self._fiber_angles = None
        self._cardiac_vectors = None
        self._cylindrical_vectors = None
        self._ellipsoidal_coordinates = None
        self._wallbounded_coordinates = None

        # Compute the eccentricity of the inner surface if not given.
        if self.parameters.get('inner_eccentricity')[1] == 0:
            V = self.parameters['cavity_volume']
            e = self._estimate_eccentricity(V)
            self.parameters['inner_eccentricity'] = e

        # Compute the eccentricity of the outer surface if not given.
        if self.parameters.get('outer_eccentricity')[1] == 0:
            V = (self.parameters['cavity_volume']
                 + self.parameters['wall_volume'])
            e = self._estimate_eccentricity(V)
            self.parameters['outer_eccentricity'] = e

        if meshfile:
            # Read the geometrical parameters from the HDF5 meshfile.
            file_extension = os.path.splitext(meshfile)[1]
            if file_extension == '.hdf5' or file_extension == '.h5':
                try:
                    # Read parameters from hdf5 file (if saved).
                    geometry_parameters = self.read_geometric_parameters(meshfile)
                    self.parameters.update(geometry_parameters)
                    print_once('Loaded saved geometry parameters from HDF5 meshfile')

                except RuntimeError as error_detail:
                    print_once('Except RuntimeError: {}'.format(error_detail))
                    print_once('No reloading of geometric parameters from meshfile.')

                # Read fiber field from file if requested and if it exists.
                load_fiber_field_from_meshfile = self.parameters['load_fiber_field_from_meshfile']
                if load_fiber_field_from_meshfile:
                    print_once('Looking for fiber field in HDF5 meshfile...')
                    self.load_fiber_field(filepath=meshfile)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this geometry.
        """
        prm = Parameters('left_ventricle')

        prm.add('wall_volume', float())
        prm.add('cavity_volume', float())

        prm.add('focus_height', float())
        prm.add('truncation_height', float())

        prm.add('inner_eccentricity', float())
        prm.add('outer_eccentricity', float())

        prm.add('mesh_segments', 20)
        prm.add('mesh_resolution', float())

        fiber_prm = Parameters('fiber_field')

        fiber_prm.add('h10', 0.3620)
        fiber_prm.add('h11', -1.1600)
        fiber_prm.add('h12', -0.1240)
        fiber_prm.add('h13', 0.1290)
        fiber_prm.add('h14', -0.0614)
        fiber_prm.add('h22', 0.0984)
        fiber_prm.add('h24', -0.0701)

        fiber_prm.add('t11', -0.6260)
        fiber_prm.add('t12', 0.5020)
        fiber_prm.add('t21', 0.6260)
        fiber_prm.add('t23', 0.2110)
        fiber_prm.add('t25', 0.0380)

        prm.add('load_fiber_field_from_meshfile', True)

        prm.add(fiber_prm)

        return prm

    def cardiac_vectors(self):
        """
        Return the associated cardiac basis vectors for this geometry.

        The order of the vectors is [ec, el, et].
        """
        if not self._cardiac_vectors:
            C = self.parameters['focus_height']
            sig, tau, phi = self.ellipsoidal_coordinates()
            ec = CoordinateSystem(CardiacBasisVectors.ec, sig, tau, phi, C)
            el = CoordinateSystem(CardiacBasisVectors.el, sig, tau, phi, C)
            et = CoordinateSystem(CardiacBasisVectors.et, sig, tau, phi, C)
            self._cardiac_vectors = ec, el, et
            # TODO Check that this ordering is correct.
        return self._cardiac_vectors

    def compute_volume(self, surface='inner', du=None):
        """
        Approximate the volume bounded by the given surface and displacements.

        Args:
            surface (optional): Specify either ``inner`` (default) or ``outer``.
            du (optional): Displacement to move mesh by before computing volume.

        Returns:
            Approximated volume of the geometry bounded by the given surface
            and displacements.
        """
        # TODO Change outer/wall computation to use native volumes.
        # TODO Figure out how to cache some of these arrays if needed.

        # noinspection PyUnusedLocal
        timer = Timer('volume computation')

        # Always make a copy of the mesh to work with.
        mesh = Mesh(self.mesh())

        # Displace the mesh if needed.
        if du:
            # Check if du is defined with linear elements. Interpolate if not.
            if du.ufl_element().degree() != 1:
                u = interpolate(du, VectorFunctionSpace(self.mesh(), 'P', 1))
            else:
                u = Function(du)
            ALE.move(mesh, u)
            del u

        # Define an integral measure over the exterior surfaces.
        ds = Measure('ds', subdomain_data=self.tags())

        # Create an array of indices where facets are on the requested surface.
        surface_ = self.endocardium if surface == 'inner' else self.epicardium
        facet_array = np.where(ds.subdomain_data().array() == surface_)[0]
        # TODO Raise error if incorrect surface is requested.

        # Create an array of the facet objects themselves.
        facet_object_array = []
        for i, j in enumerate(facets(mesh)):
            if i in facet_array:
                facet_object_array.append(j)

        # Create an array of the nodal coordinates of these facets.
        x = []
        for i in facet_object_array:
            for j in vertices(i):
                x.append(j.point().array()[:])
        x = np.array(x)

        # Compute the volume as an approximation to [0, 0, h].
        volume = 0
        d = np.array([0, 0, self.parameters['truncation_height']])
        for i in range(0, len(x), 3):
            a = x[i + 0] - d
            b = x[i + 1] - d
            c = x[i + 2] - d
            volume += np.abs(np.dot(a, np.cross(b, c))/6)

        # Delete the copied mesh.
        del mesh

        # Gather up the volumes to a single process and return.
        return MPI.sum(mpi_comm_world(), volume)

    def cylindrical_vectors(self):
        """
        Return the associated cylindrical basis vectors for this geometry.

        The order of the vectors is [ez, er, ec].
        """
        if not self._cylindrical_vectors:
            ez = CoordinateSystem(CylindricalBasisVectors.ez)
            er = CoordinateSystem(CylindricalBasisVectors.er)
            ec = CoordinateSystem(CylindricalBasisVectors.ec, ez, er)
            self._cylindrical_vectors = ez, er, ec
        return self._cylindrical_vectors

    def ellipsoidal_coordinates(self):
        """
        Return the associated ellipsoidal coordinates for this geometry.

        The order of the coordinates is [σ, τ, φ].
        """
        if not self._ellipsoidal_coordinates:
            C = self.parameters['focus_height']
            sig = CoordinateSystem(EllipsoidalCoordinates.sigma, C)
            tau = CoordinateSystem(EllipsoidalCoordinates.tau, C)
            phi = CoordinateSystem(EllipsoidalCoordinates.phi)
            self._ellipsoidal_coordinates = sig, tau, phi
        return self._ellipsoidal_coordinates

    def fiber_angles(self):
        """
        Return the associated fiber angles for this geometry.

        The order of the angles is [ah, at].
        """
        if not self._fiber_angles:
            h10 = self.parameters['fiber_field']['h10']
            h11 = self.parameters['fiber_field']['h11']
            h12 = self.parameters['fiber_field']['h12']
            h13 = self.parameters['fiber_field']['h13']
            h14 = self.parameters['fiber_field']['h14']
            h22 = self.parameters['fiber_field']['h22']
            h24 = self.parameters['fiber_field']['h24']
            t11 = self.parameters['fiber_field']['t11']
            t12 = self.parameters['fiber_field']['t12']
            t21 = self.parameters['fiber_field']['t21']
            t23 = self.parameters['fiber_field']['t23']
            t25 = self.parameters['fiber_field']['t25']
            u, v = self.wallbounded_coordinates()
            ah_, at_ = DefaultFiberAngles.helix, DefaultFiberAngles.transverse
            ah = CoordinateSystem(ah_, u, v, h10, h11, h12, h13, h14, h22, h24)
            at = CoordinateSystem(at_, u, v, t11, t12, t21, t23, t25)
            self._fiber_angles = ah, at
        return self._fiber_angles

    def wallbounded_coordinates(self):
        """
        Return the associated wall-bounded normalized ellipsoidal coordinates
        for this geometry.

        The order of the coordinates is [u, v].
        """
        if not self._wallbounded_coordinates:
            h = self.parameters['truncation_height']
            sig1 = 1/self.parameters['inner_eccentricity']
            sig2 = 1/self.parameters['outer_eccentricity']
            sig, tau, phi = self.ellipsoidal_coordinates()
            u = CoordinateSystem(WallBoundedCoordinates.u, sig, tau, h)
            v = CoordinateSystem(WallBoundedCoordinates.v, sig, tau, sig1, sig2)
            self._wallbounded_coordinates = u, v
        return self._wallbounded_coordinates

    @property
    def base(self):
        """
        Return the surface number of the base.
        """
        return 1

    @property
    def endocardium(self):
        """
        Return the surface number of the endocardium (inner wall).
        """
        return 2

    @property
    def epicardium(self):
        """
        Return the surface number of the epicardium (outer wall).
        """
        return 3

    @staticmethod
    def read_geometric_parameters(meshfile):
        """
        Reads geometric parameters from HDF5 file.

        Args:
            meshfile (str): Path to HDF5 file that contains the geometric parameters
                            (as saved with the function save_mesh_to_hdf5).

        Returns:
            A dolfin parameter set with the geometric parameters
            (radii of ellipsoids and truncation height).
        """
        # Create parameter set.
        geometry = Parameters('left_ventricle')
        # Read HDF5 file.
        with HDF5File(mpi_comm_world(), meshfile, 'r') as f:
            for k, v in f.attributes('geometry').items():
                # Collect the the geometric parameters in the parameter set.
                if k != 'load_fiber_field_from_meshfile':
                    geometry.add(k, v)
        return geometry

    def save_mesh_to_hdf5(self, meshfile, save_fiber_vector=True, V=None):
        """
        Saves mesh, geometric parameters and fiber field to an HDF5 file,
        that can be loaded for re-use with FEniCS.

        Args:
            meshfile (str): Name or relative/full path of HDF5 file to write to.
            save_fiber_vector (bool): Saves fiber vector to same hdf5 file (True) or do not save (False).
            V (dolfin vector function space): Vector function space to define the fiber vectors on (optional)
                                              This is not needed if save_fiber_vector is False,
                                              or if the fiber vectors have already been created.

            Note: if save_fiber_vector is True and V is None,
            the function object of ef should've already been created
            before calling this function.
        """
        # Write the mesh using HDF5File from FEniCS.
        with HDF5File(mpi_comm_world(), meshfile, 'w') as f:
            f.write(self.mesh(), 'mesh')

            # Write the fiber vector if requested.
            if save_fiber_vector:
                f.write(self.fiber_vectors()[0].to_function(V), 'fiber_vector')

        # Add the geometric parameters to the same HDF5 file.
        if MPI.rank(mpi_comm_world()) == 0:
            with HDF5File(mpi_comm_self(), meshfile, 'a') as f:
                geometric_parameters = self.parameters
                # Create (empty) geometry dataset in hdf5 file.
                f.write(np.array([], dtype=np.float_), 'geometry')
                # Add geometry parameters as attributes to the geometry dataset.
                geo_data = f.attributes('geometry')
                for k, v in geometric_parameters.items():
                    # We do not need to save the fiber field parameters.
                    if k != 'fiber_field':
                        geo_data[k] = v

    def _create_fiber_vectors(self):
        """
        Helper method to create the fiber (ef, es, en) basis vectors.
        """
        ah, at = self.fiber_angles()
        ec, el, et = self.cardiac_vectors()
        ef = CoordinateSystem(FiberBasisVectors.ef, ah, at, ec, el, et)
        es = CoordinateSystem(FiberBasisVectors.es, at, ec, et)
        en = CoordinateSystem(FiberBasisVectors.en, ef, es)
        return ef, es, en

    def _create_mesh(self):
        """
        Helper method to interface with :class:`~cvbtk.LeftVentricleMesh`.
        """
        C = self.parameters['focus_height']
        h = self.parameters['truncation_height']
        e1 = self.parameters['inner_eccentricity']
        e2 = self.parameters['outer_eccentricity']
        resolution = self.parameters['mesh_resolution']
        segments = self.parameters['mesh_segments']
        return LeftVentricleMesh(C, h, e1, e2, resolution, segments=segments)

    def _create_tags(self):
        """
        Helper method to mark the basal, inner, and outer surface boundaries.
        """
        # Initialize (create) the volume mesh and boundary (surface) mesh.
        mesh = self.mesh()
        b_mesh = BoundaryMesh(mesh, 'exterior')

        # Initialize the boundary mesh and create arrays of boundary faces.
        b_map = b_mesh.entity_map(2)
        b_faces = [Facet(mesh, b_map[cell.index()]) for cell in cells(b_mesh)]

        # Create an empty list of marker values to fill in.
        b_value = np.zeros(len(b_faces), dtype=int)

        # Compute the midpoint locations of each facet on the boundary.
        X = np.array([each.midpoint() for each in b_faces])

        # Compute the sigma value of each facet using the midpoint coordinates.
        C = self.parameters['focus_height']
        sig = 0.5/C*(np.sqrt(X[:, 0]**2 + X[:, 1]**2 + (X[:, 2] + C)**2)
                     + np.sqrt(X[:, 0]**2 + X[:, 1]**2 + (X[:, 2] - C)**2))

        # Fill in the marker values using sigma in relation to the mean values.
        # noinspection PyUnresolvedReferences
        b_value[sig > sig.mean()] = self.epicardium
        # noinspection PyUnresolvedReferences
        b_value[sig < sig.mean()] = self.endocardium

        # The base is where the z-normal is vertical. Must be called last.
        n_z = np.array([each.normal()[2] for each in b_faces])
        b_value[n_z == 1] = self.base

        # Create a FacetFunction and initialize to zero.
        marker = FacetFunction('size_t', mesh, value=0)

        # Fill in the FacetFunction with the marked values and return.
        for f, v in zip(b_faces, b_value):
            marker[f] = v
        return marker

    def _estimate_eccentricity(self, target_volume):
        """
        Estimate the eccentricity required for a truncated ellipsoid to meet the
        given target volume.
        """
        C = self.parameters['focus_height']
        h = self.parameters['truncation_height']

        # Define a function to give to SciPy's Newton solver to obtain e.
        def func(x):
            y = DOLFIN_PI*(1 - x**2)*(2/3*(C/x)**3 + h*(C/x)**2 - 1/3*h**3)
            return y - target_volume

        # Compute (estimate) and return the required eccentricity.
        return scipy.optimize.newton(func, 0.9)


class BiventricleGeometry(BaseGeometry):
    """
    High-level interface to a biventricular heart model for simulation and post-processing.

    Args:
        meshfile (optional): Path to an existing mesh to load in.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, meshfile=None, **kwargs):
        super(BiventricleGeometry, self).__init__(meshfile=meshfile, **kwargs)

        self._fiber_angles = None
        self._cardiac_vectors = None
        self._cylindrical_vectors = None
        self._ellipsoidal_coordinates = None
        self._wallbounded_coordinates = None
        self._unit_vectors_bayer = None
        self._volume_tags = None

        if meshfile:
            # Read the geometrical parameters from the HDF5 meshfile.
            file_extension = os.path.splitext(meshfile)[1]
            if file_extension == '.hdf5' or file_extension == '.h5':
                # Read parameters from hdf5 file.
                print_once('Loading saved geometry parameters from HDF5 meshfile...')
                geometry_parameters = self.read_geometric_parameters(meshfile)

                # Read fiber field from file if requested and if it exists.
                load_fiber_field_from_meshfile = self.parameters['load_fiber_field_from_meshfile']
                if load_fiber_field_from_meshfile:
                    print_once('Looking for fiber field in HDF5 meshfile...')
                    self.load_fiber_field(filepath=meshfile)
            else:
                geometry_parameters = self.default_geometry_parameters()
                if 'geometry' in kwargs.keys():
                    print_once('Using specified geometry parameters...')
                    geometry_parameters.update(kwargs['geometry'])
                else:
                    print_once('Using default geometry parameters...')
        else:
            geometry_parameters = compute_biventricular_parameters(self.parameters)

        self.parameters.add(geometry_parameters)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this geometry.
        """
        prm = Parameters('biventricle')
        prm.add('f_R', float())
        prm.add('f_V', float())
        prm.add('f_h1', float())
        prm.add('f_T', float())
        prm.add('f_sep', float())
        prm.add('f_VRL', float())
        prm.add('V_lvw', float())
        prm.add('theta_A', float())

        prm.add('mesh_segments', int())
        prm.add('mesh_resolution', float())

        # Mesh generator settings
        mesh_generator_parameters = Parameters('mesh_generator')
        mesh_generator_parameters.add('mesh_resolution', 30.)
        mesh_generator_parameters.add('feature_threshold', 70.)
        mesh_generator_parameters.add('odt_optimize', False)
        mesh_generator_parameters.add('lloyd_optimize', False)
        mesh_generator_parameters.add('perturb_optimize', False)
        mesh_generator_parameters.add('exude_optimize', False)

        prm.add(mesh_generator_parameters)

        # Fiber field.
        fiber_prm = Parameters('fiber_field')
        fiber_prm.add('h10', 0.3620)
        fiber_prm.add('h11', -1.1600)
        fiber_prm.add('h12', -0.1240)
        fiber_prm.add('h13', 0.1290)
        fiber_prm.add('h14', -0.0614)
        fiber_prm.add('h22', 0.0984)
        fiber_prm.add('h24', -0.0701)
        fiber_prm.add('t11', -0.6260)
        fiber_prm.add('t12', 0.5020)
        fiber_prm.add('t21', 0.6260)
        fiber_prm.add('t23', 0.2110)
        fiber_prm.add('t25', 0.0380)

        prm.add(fiber_prm)

        prm.add('load_fiber_field_from_meshfile', True)

        # Bayer unit vectors.
        bayer_prm = BasisVectorsBayer.default_parameters()
        prm.add(bayer_prm)

        return prm

    @staticmethod
    def default_geometry_parameters():
        prm = Parameters('geometry')
        prm.add('R_1', 2.54843)
        prm.add('R_1sep', 1.7839)
        prm.add('R_2', 4.18024)
        prm.add('R_2sep', 3.41571)
        prm.add('R_3', 6.69962)
        prm.add('R_3y', 3.64174)
        prm.add('R_4', 7.23812)
        prm.add('Z_1', 3.30965)
        prm.add('Z_2', 4.68334)
        prm.add('Z_3', 3.73358)
        prm.add('Z_4', 4.27208)
        prm.add('h', 1.32386)
        return prm

    def _fiber_angle_functions(self):
        """
        Return the associated functions for the fiber angles for this geometry.

        Note that the returned objects are Pyhton functions that are only functions
        of a set of normalized longitudinal (u) and transversal (v) coordinates.

        The order of the functions is [ah(u,v), at(u,v)].
        """
        h10 = self.parameters['fiber_field']['h10']
        h11 = self.parameters['fiber_field']['h11']
        h12 = self.parameters['fiber_field']['h12']
        h13 = self.parameters['fiber_field']['h13']
        h14 = self.parameters['fiber_field']['h14']
        h22 = self.parameters['fiber_field']['h22']
        h24 = self.parameters['fiber_field']['h24']
        t11 = self.parameters['fiber_field']['t11']
        t12 = self.parameters['fiber_field']['t12']
        t21 = self.parameters['fiber_field']['t21']
        t23 = self.parameters['fiber_field']['t23']
        t25 = self.parameters['fiber_field']['t25']

        ah_, at_ = DefaultFiberAngles.ah_single, DefaultFiberAngles.at_single

        # Fix input parameters.
        ah = functools.partial(ah_, h10=h10, h11=h11, h12=h12, h13=h13, h14=h14, h22=h22, h24=h24)
        at = functools.partial(at_, t11=t11, t12=t12, t21=t21, t23=t23, t25=t25)

        return ah, at

    @property
    def unit_vectors_bayer(self):
        """
        Returns the BasisVectorsBayer object.
        """
        if not self._unit_vectors_bayer:
            self._create_unit_vectors_bayer()
        return self._unit_vectors_bayer

    def _create_unit_vectors_bayer(self):
        """
        Creates object that contains normalized coordinates,
        cardiac vector basis and fiber vectors for this geometry.
        """
        # Functions for helix and transverse angles (must only be functions of (u, v))
        ah, at = self._fiber_angle_functions()
        self._unit_vectors_bayer = BasisVectorsBayer(self, func_ah_w=ah, func_at=at, **self.parameters['bayer'])

    def cardiac_vectors(self):
        """
        Return the associated cardiac basis vectors for this geometry.

        The order of the vectors is [ec, el, et].
        """
        if not self._cardiac_vectors:
            ec = CoordinateSystem(self.unit_vectors_bayer.ec)
            el = CoordinateSystem(self.unit_vectors_bayer.el)
            et = CoordinateSystem(self.unit_vectors_bayer.et)
            self._cardiac_vectors = ec, el, et
        return self._cardiac_vectors

    def cylindrical_vectors(self):
        """
        Return the associated cylindrical basis vectors for this geometry as UFL expressions.

        The order of the vectors is [ez, er, ec].
        """
        if not self._cylindrical_vectors:

            # TODO create CoordinateSystems for cylindrical vectors instead of UFL expressions
            # (prevents the need for et to be defined).

            # Get et.
            et = self.cardiac_vectors()[2]
            try:
                if et._function is None:

                    # Discretize the fiber vectors onto quadrature elements.
                    mesh = self.mesh()

                    # Check if quadrature degree is set in parameters.
                    if parameters['form_compiler']['quadrature_degree'] is None:
                        raise RuntimeError(
                            "Quadrature degree not specified in parameters['form_compiler']['quadrature_degree']")

                    # Create quadrature function space.
                    V_q = quadrature_function_space(mesh)

                    # Compute fiber vectors at quadrature points.
                    et = et.to_function(V_q)

            except AttributeError:
                # Fiber vectors are UFL-like definitions that do not need to be discretized.
                pass

            ez = CylindricalBasisVectorsBiV.ez()
            er = CylindricalBasisVectorsBiV.er(et, ez)
            ec = CylindricalBasisVectorsBiV.ec(ez, er)
            self._cylindrical_vectors = ez, er, ec

        return self._cylindrical_vectors

    def wallbounded_coordinates(self):
        """
        Return the associated wall-bounded normalized ellipsoidal coordinates
        for this geometry.

        The order of the coordinates is [u, v_wall, v_septum].
        """
        if not self._wallbounded_coordinates:
            u = CoordinateSystem(self.unit_vectors_bayer.u)
            v_wall = CoordinateSystem(self.unit_vectors_bayer.v_wall)
            v_septum = CoordinateSystem(self.unit_vectors_bayer.v_septum)
            self._wallbounded_coordinates = u, v_wall, v_septum
        return self._wallbounded_coordinates

    def _create_fiber_vectors(self):
        """
        Helper method to create the fiber (ef, es, en) basis vectors.
        """
        ef = CoordinateSystem(self.unit_vectors_bayer.ef)
        es = CoordinateSystem(self.unit_vectors_bayer.es)
        en = CoordinateSystem(self.unit_vectors_bayer.en)
        return ef, es, en

    def fiber_angles(self):
        """
        Return the effective fiber angles for this geometry.
        Only used for visualisation.

        The order of the angles is [ah, at].
        """
        if not self._fiber_angles:
            ah_, at_ = ComputeFiberAngles.ah, ComputeFiberAngles.at
            ah = CoordinateSystem(ah_, self.cardiac_vectors(), self.fiber_vectors())
            at = CoordinateSystem(at_, self.cardiac_vectors(), self.fiber_vectors())
            self._fiber_angles = ah, at
        return self._fiber_angles

    def compute_volume_from_surface(self, boundary_tag, mesh=None):
        """
        Approximate the volume bounded by the given surface marked with boundary_tag.

        Args:
            boundary_tag: Specify the surface from which to approximate the volume.
            mesh (optional): Specify a mesh or use the current mesh (default).

        Returns:
            Approximated volume of the geometry bounded by the given surface.
        """
        if not mesh:
            mesh = Mesh(self.mesh())

        # Define an integral measure over the exterior surfaces.
        ds = Measure('ds', subdomain_data=self.tags())

        facet_array = np.where(ds.subdomain_data().array() == boundary_tag)[0]

        # Create an array of the facet objects themselves.
        facet_object_array = []
        for i, j in enumerate(facets(mesh)):
            if i in facet_array:
                facet_object_array.append(j)

        # Create an array of the nodal coordinates of these facets.
        x = []
        for i in facet_object_array:
            for j in vertices(i):
                x.append(j.point().array()[:])
        x = np.array(x)

        # Compute the volume as an approximation to [0, 0, h].
        volume = 0
        # Introduce a sum to volume to ensure that the current node has at least 1 summation to perform in case len(x) is zero.
        # If not introduced while len(x) is 0, the gathering at the end of this function will be waiting forever for all the nodes to complete their summation.
        volume += 0.
        h = self.parameters['geometry']['h']
        d = np.array([0, 0, h])
        for i in range(0, len(x), 3):
            a = x[i + 0] - d
            b = x[i + 1] - d
            c = x[i + 2] - d
            volume += np.abs(np.dot(a, np.cross(b, c)) / 6)

        # Gather up the volumes to a single process and return.
        return MPI.sum(mpi_comm_world(), volume)

    def compute_volume_from_cells(self, volume_tag, mesh=None):
        """
        Approximate the volume bounded by the given surface marked with boundary_tag.

        Args:
            volume_tag: Specify the part of the mesh from which to approximate the volume.
            mesh (optional): Specify a mesh or use the current mesh (default).

        Returns:
            Approximated volume of the geometry bounded by the given surface.
        """
        if not mesh:
            mesh = Mesh(self.mesh())

        # Define an integral measure over the exterior surfaces.
        dcells = Measure('cell', subdomain_data=self.volume_tags())

        cell_array = np.where(dcells.subdomain_data().array() == volume_tag)[0]

        # Add the volumes of the specified cells.
        volume = 0
        # Introduce a sum to volume to ensure that the current node has at least 1 summation to perform in case len(cell_array) is zero.
        # If not introduced while len(cell_array) is 0, the gathering at the end of this function will be waiting forever for all the nodes to complete their summation.
        volume += 0.
        for i, j in enumerate(cells(mesh)):
            if i in cell_array:
                volume += j.volume()

        # Gather up the volumes to a single process and return.
        return MPI.sum(mpi_comm_world(), volume)

    def compute_volume(self, ventricle, volume='cavity', du=None):
        """
        Approximate the volume bounded by the given surface and displacements.

        Args:
            ventricle: Specify either 'lv' or 'rv'.
            volume (optional): Specify either ``cavity`` (default) or ``wall``.
            du (optional): Displacement to move mesh by before computing volume.

        Returns:
            Approximated volume of the given part of the (displaced) mesh.
        """

        # Always make a copy of the mesh to work with.
        mesh = Mesh(self.mesh())

        # Displace the mesh if needed.
        if du:
            # Check if du is defined with linear elements. Interpolate if not.
            if du.ufl_element().degree() != 1:
                u = interpolate(du, VectorFunctionSpace(self.mesh(), 'P', 1))
            else:
                u = Function(du)
            ALE.move(mesh, u)
            del u

        # Volume computations from relevant surfaces.
        if volume is 'wall':
            if ventricle is 'lv':
                volume = self.compute_volume_from_cells(self.lv_tag, mesh=mesh)
            elif ventricle is 'rv':
                volume = self.compute_volume_from_cells(self.rv_tag, mesh=mesh)
            else:
                raise ValueError("Given ventricle argument '{}' in compute_volume is invalid. Choose either '{}' or '{}'.".format(ventricle, 'lv', 'rv'))

        elif volume is 'cavity':
            if ventricle is 'lv':
                volume = self.compute_volume_from_surface(self.lv_endocardium, mesh=mesh)
            elif ventricle is 'rv':
                vol1 = self.compute_volume_from_surface(self.rv_endocardium, mesh=mesh)
                vol2 = self.compute_volume_from_surface(self.rv_septum, mesh=mesh)
                volume = vol1 - vol2
            else:
                raise ValueError(
                    "Given ventricle argument '{}' in compute_volume is invalid. Choose either '{}' or '{}'.".format(
                        ventricle, 'lv', 'rv'))

        else:
            raise ValueError(
                "Given volume argument '{}' in compute_volume is invalid. Choose either '{}' or '{}'.".format(
                    volume, 'cavity', 'wall'))

        # Delete the copied mesh.
        del mesh

        return volume


    @property
    def base(self):
        """
        Return the surface number of the base.
        """
        return 1

    @property
    def lv_endocardium(self):
        """
        Return the surface number of the LV endocardium (inner wall).
        """
        return 2

    @property
    def lv_epicardium(self):
        """
        Return the surface number of the LV epicardium (outer wall).
        """
        return 3

    @property
    def rv_endocardium(self):
        """
        Return the surface number of the RV endocardium belonging to the RV free wall.
        """
        return 4

    @property
    def rv_septum(self):
        """
        Return the surface number of the RV endocardium belonging to the septal wall.
        """
        return 5

    @property
    def rv_epicardium(self):
        """
        Return the surface number of the RV epicardium (outer wall).
        """
        return 6

    @property
    def lv_tag(self):
        """
        Return the volume number (marker) of the LV.
        """
        return 1

    @property
    def rv_tag(self):
        """
        Return the volume number (marker) of the RV.
        """
        return 2

    def volume_tags(self):
        """
        Return the associated :class:`~dolfin.CellFunction` for this geometry.
        """
        if not self._volume_tags:
            self._volume_tags = self._create_volume_tags()
        return self._volume_tags

    def _create_mesh(self):
        """
        Helper method to create the mesh.
        """
        # Gather the relevant optional arguments.
        geometry_parameters = self.parameters['geometry']
        segments = self.parameters['mesh_segments']
        mesh_generator_parameters = self.parameters['mesh_generator']
        mesh = BiventricleMesh(geometry_parameters, segments=segments,
                                     mesh_generator_parameters=mesh_generator_parameters)
        return mesh

    def _create_tags(self, return_shape_error = False):
        """
        Helper function to create boundary tags (markers) for boundary facets.
        The boundary will be classified as one of the following:
            LV endocardium
            LV epicardium free wall
            RV endocardium free wall
            RV epicardium free wall
            RV endocardium septal wall
            Base

        Args:
            return_shape_error (boolean): Returns a facet function with the maximum distance to the analytical shape if True.
                                          Returns a facet function with the tags/markers if False (default).
        """

        def dist_to_ellipsoid(coord, center, R, R_y, Z):
            """
            Calculates the closest distance of a point with coordinates coord to the ellipsoid given by
            axial dimensions R, R_y and Z with the center point center.

            Args:
                coord (2d array): The x, y and z coordinates of one or more points with shape (n_points, 3).
                center (tuple): The x, y and z coordinates of the center of the ellipsoid.
                R (float): semi-principal axis of ellipsoid in x direction.
                R_y (float): semi-principal axis of ellipsoid in y direction.
                Z (float): semi-principal axis of ellipsoid in z direction.

            Returns:
                The closest distance of a point with coordinates coord to the ellipsoid.
            """
            # Extract x, y and z coordinates of all given points
            x = coord[:, 0]
            y = coord[:, 1]
            z = coord[:, 2]

            # Compute ellipsoidal coordinates phi and theta, assuming x, y, z is on the ellipsoid.
            phi = np.arctan2((y - center[1]) * R, (x - center[0]) * R_y)
            theta = np.arcsin(np.clip((z - center[2]) / Z, -1.,
                                      1.))  # Note clipping the argument of arcsin, which may be needed when z does not lie (exactly) on the ellipsoid.

            # Compute x, y and z coordinates of the point on the ellipsoid corresponding to phi and theta.
            x_c = R * np.cos(theta) * np.cos(phi) + center[0]
            y_c = R_y * np.cos(theta) * np.sin(phi) + center[1]
            z_c = Z * np.sin(theta) + center[2]

            # Compute the distance of the computed x-, y-, z-coordinates (that lie on the eelipsoid) with the actual coordinates.
            # Return the maximal distance found.
            return np.max(np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2 + (z - z_c) ** 2))

        def dist_to_lv_fw(coord, geometry_parameters, surface='endo'):
            """"
            Args:
                coord (2d array): The x, y and z coordinates of one or more points with shape (n_points, 3).
                surface (str): 'endo' or 'epi' ; specifies which boundary to compare the point(s) in coord to.
            Returns:
                The closest distance of a point with coordinates coord to the LV free wall.
            """

            if np.mean(coord[:, 0]) < 0:
                # Not part of LV free wall
                return np.inf

            else:
                if surface == 'endo':
                    R = geometry_parameters['R_1']
                    R_y = geometry_parameters['R_1']
                    Z = geometry_parameters['Z_1']
                elif surface == 'epi':
                    R = geometry_parameters['R_2']
                    R_y = geometry_parameters['R_2']
                    Z = geometry_parameters['Z_2']
                else:
                    raise NotImplementedError

                return dist_to_ellipsoid(coord, (0, 0, 0), R, R_y, Z)

        def dist_to_lv_sw(coord, geometry_parameters, surface='endo'):
            """"
            Args:
                coord (2d array): The x, y and z coordinates of one or more points with shape (n_points, 3).
                surface (str): 'endo' or 'epi' ; specifies which boundary to compare the point(s) in coord to.
            Returns:
                The closest distance of a point with coordinates coord to the LV septal wall.
            """

            if np.mean(coord[:, 0]) > 0:
                # Not part of septal wall
                return np.inf

            else:
                if surface == 'endo':
                    R = geometry_parameters['R_1sep']
                    R_y = geometry_parameters['R_1']
                    Z = geometry_parameters['Z_1']
                elif surface == 'epi':
                    R = geometry_parameters['R_2sep']
                    R_y = geometry_parameters['R_2']
                    Z = geometry_parameters['Z_2']
                else:
                    raise NotImplementedError

                return dist_to_ellipsoid(coord, (0, 0, 0), R, R_y, Z)

        def dist_to_rv_fw(coord, geometry_parameters, surface='endo'):
            """"
            Args:
                coord (2d array): The x, y and z coordinates of one or more points with shape (n_points, 3).
                surface (str): 'endo' or 'epi' ; specifies which boundary to compare the point(s) in coord to.
            Returns:
                The closest distance of a point with coordinates coord to the RV free wall.
            """
            if np.mean(coord[:, 0]) > 0.:
                # Not part of rv wall
                return np.inf

            else:
                if surface == 'endo':
                    R = geometry_parameters['R_3']
                    R_y = geometry_parameters['R_3y']
                    Z = geometry_parameters['Z_3']
                elif surface == 'epi':
                    R = geometry_parameters['R_4']
                    R_y = geometry_parameters['R_2']
                    Z = geometry_parameters['Z_4']
                else:
                    raise NotImplementedError

                center_rv_x = 0.  # center of RV is at origin.

                return dist_to_ellipsoid(coord, (center_rv_x, 0, 0), R, R_y, Z)

        def compute_x_att_endo(z, geometry_parameters):
            """
            Computes the x-coordinate where the LV epicardium and RV endocardium attach.
            NOTE: only works when center of LV is (0, 0, 0).
            :return: x_att_endo
            """

            # Extract geometric parameters.
            R_2 = geometry_parameters['R_2']
            R_2sep = geometry_parameters['R_2sep']
            R_3 = geometry_parameters['R_3']
            R_3y = geometry_parameters['R_3y']
            Z_2 = geometry_parameters['Z_2']
            Z_3 = geometry_parameters['Z_3']

            # Compute d_1 and d_2
            d_1 = 1 - (z / Z_2) ** 2
            d_2 = 1 - (z / Z_3) ** 2

            # Solve abc forumula.
            center_rv_x = 0. # center of RV is at origin.
            a = 1 / (R_3 ** 2) - (R_2 / (R_2sep * R_3y)) ** 2
            b = -2 * center_rv_x / (R_3 ** 2)
            c = (R_2 / R_3y) ** 2 * d_1 - d_2 + (center_rv_x / R_3) ** 2

            nom1 = -b + np.sqrt(b ** 2 - 4 * a * c)
            nom2 = -b - np.sqrt(b ** 2 - 4 * a * c)
            den = 2 * a

            # Return lowest x value
            return np.min((nom1 / den, nom2 / den))

        def compute_x_att_epi(z, geometry_parameters):
            """
            Computes the x-coordinate where the LV epicardium and RV epicardium attach.
            NOTE: only works when center of LV is (0, 0, 0).
            :return: x_att_epi
            """

            # Extract geometric parameters.
            R_2 = geometry_parameters['R_2']
            R_2sep = geometry_parameters['R_2sep']
            R_4 = geometry_parameters['R_4']
            Z_2 = geometry_parameters['Z_2']
            Z_4 = geometry_parameters['Z_4']

            # Compute d_1 and d_2
            d_1 = 1 - (z / Z_2) ** 2
            d_2 = 1 - (z / Z_4) ** 2

            # Solve abc forumula.
            center_rv_x = 0. # center of RV is at origin.
            a = 1 / (R_4 ** 2) - (R_2 / (R_2sep * R_2)) ** 2
            b = -2 * center_rv_x / (R_4 ** 2)
            c = (R_2 / R_2) ** 2 * d_1 - d_2 + (center_rv_x / R_4) ** 2

            nom1 = -b + np.sqrt(b ** 2 - 4 * a * c)
            nom2 = -b - np.sqrt(b ** 2 - 4 * a * c)
            den = 2 * a

            # Return lowest x value
            return np.min((nom1 / den, nom2 / den))

        # Extract geometry parameters.
        geometry_parameters = self.parameters['geometry']

        b_mesh = BoundaryMesh(self.mesh(), 'exterior')

        # Initialize the boundary mesh and create arrays of boundary faces.
        b_map = b_mesh.entity_map(2)
        b_faces = [Facet(self.mesh(), b_map[cell.index()]) for cell in cells(b_mesh)]

        # Create an empty list of marker values to fill in.
        b_value = np.zeros(len(b_faces), dtype=int)
        error_shape = np.zeros(len(b_faces), dtype=float)

        # Collect all markers in an ordered list.
        boundary_markers = [self.lv_endocardium,
                            self.lv_epicardium,
                            self.rv_endocardium,
                            self.rv_epicardium,
                            self.rv_septum]

        # Determine the marker value based on coordinates.
        for idx, facet in enumerate(b_faces):

            # We can either use the midpoint of the facet, or the investigate all the vertices of the facet.
            # # Compute the midpoint locations of the facet on the boundary.
            # coord = facet.midpoint()

            # Create an array of the nodal coordinates of the facet.
            coord = []
            for j in vertices(facet):
                coord.append(j.point().array()[:])

            # Make sure coord is a 2d array (important when coord holds only one point's coordinates).
            coord = np.reshape(coord, (-1, 3))

            # Compute distance(s) of the point(s) to every ellipsoidal part of the biventricular mesh. Choose the maximal distance of the points in coord.
            distance_to_boundaries = np.zeros(5, dtype=float)+np.inf # 5 boundaries (excl. base): (LV, RV) x (endo, epi) and RV septum

            # LV endocardium
            distance_to_boundaries[0] = np.min(np.abs([dist_to_lv_fw(coord, geometry_parameters, surface='endo'), dist_to_lv_sw(coord, geometry_parameters, surface='endo')]))

            # LV epicardium (free wall)
            distance_to_boundaries[1] = dist_to_lv_fw(coord, geometry_parameters, surface='epi')

            # RV endocardium (free wall)
            distance_to_boundaries[2] = dist_to_rv_fw(coord, geometry_parameters, surface='endo')

            # RV epicardium (free wall)
            distance_to_boundaries[3] = dist_to_rv_fw(coord, geometry_parameters, surface='epi')

            # RV septum (actually a part of the LV epicardium)
            # Compute attachment of LV epicardium and RV endocardium at current z coordinate.
            z = np.mean(coord[:,2])
            x_att_endo = compute_x_att_endo(z, geometry_parameters)

            # Compute attachment of LV epicardium and RV epicardium at current z coordinate.
            x_att_epi = compute_x_att_epi(z, geometry_parameters)

            # Compute x coordinate that splits RV septal wall from LV epicardium free wall.
            x_septum = np.mean((x_att_endo, x_att_epi))

            if np.mean(coord[:,0]) < x_septum:
                # Boundary facet may belong to RV septum.
                distance_to_boundaries[4] = dist_to_lv_sw(coord, geometry_parameters, surface='epi')
            else:
                # Boundary facet may belong to LV epicardium of the free wall.
                distance_to_boundaries[1] = np.min(np.abs([dist_to_lv_sw(coord, geometry_parameters, surface='epi'), distance_to_boundaries[1]]))

            # Determine to which ellipsoidal part the current facet is closest to. This determines the tag/mark
            b_value[idx] = boundary_markers[np.argmin(np.abs(distance_to_boundaries))]

            # Also keep track of the deviation from the analytical shape.
            error_shape[idx] = np.min(np.abs(distance_to_boundaries))

        # The base is where the z-normal is vertical. Must be called last.
        n_z = np.array([each.normal()[2] for each in b_faces])
        b_value[n_z == 1] = self.base
        # Set error to 0 at these facets
        error_shape[n_z == 1] = 0.

        if return_shape_error:
            # Display the error of the shape with respect to the analytical shape.
            print_once('Mean distance/Z_2 of element vertices to analytical shape: {}'.format(
                np.mean(error_shape) / geometry_parameters['Z_2']))
            print_once('Maximum distance/Z_2 of element vertices to analytical shape: {}'.format(
                np.max(error_shape) / geometry_parameters['Z_2']))

            # Create a FacetFunction and initialize to zero.
            error = FacetFunction('double', self.mesh(), value=0)

            # Fill in the FacetFunction with the error values and return.
            for f, v in zip(b_faces, error_shape):
                error[f] = v
            return error
        else:
            # Create a FacetFunction and initialize to zero.
            tags = FacetFunction('size_t', self.mesh(), value=0)

            # Fill in the FacetFunction with the marked values and return.
            for f, v in zip(b_faces, b_value):
                tags[f] = v
            return tags

    def _create_volume_tags(self):
        """
        Helper function to create volume tags (markers) for every cell.
        The cells will be classified as either LV or RV.
        """
        # Extract relevant geometric parameters.
        R_2 = self.parameters['geometry']['R_2']
        R_2sep = self.parameters['geometry']['R_2sep']
        Z_2 = self.parameters['geometry']['Z_2']

        # Collect all the cells in the mesh.
        m_cells = [Cell(self.mesh(), idx) for idx in range(self.mesh().num_cells())]

        # Compute the midpoint locations of each cell in the mesh.
        X = np.array([each.midpoint() for each in m_cells])

        # Create an empty list of marker values to fill in.
        m_value = np.zeros(len(X), dtype=int)

        # Determine the marker value based on x-coordinate.
        for idx, coord in enumerate(X):
            x, y, z = coord

            # Compute (negative) x-coordinate of septal ellipsoid at y=y and z=z.
            # (x/a)**2 = (1 - (y/b)**2 - (z/c)**2) with a = R_2sep, b = R_2 and c = Z_2

            # y and z should not exceed the principal axis of the ellipsoid.
            y_clipped = np.clip(y, -R_2, R_2)
            z_clipped = np.clip(z, -Z_2, Z_2)

            # Compute (negative) x-coordinate of septal ellipsoid.
            x_septum = -np.sqrt(R_2sep ** 2 * (1 - (y_clipped / R_2) ** 2 - (z_clipped / Z_2) ** 2))

            # If x is on the left of x_septum, assign to RV
            if x < x_septum:
                m_value[idx] = self.rv_tag
            else:
                m_value[idx] = self.lv_tag

        # Create a CellFunction and initialize to zero.
        tags = CellFunction('size_t', self.mesh(), value=0)

        # Fill in the FacetFunction with the marked values and return.
        for f, v in zip(m_cells, m_value):
            tags[f] = v

        # print_once('Material tags created in {} s.'.format(time.time()-t0))
        return tags

    def save_mesh_to_hdf5(self, meshfile, save_fiber_vector=True, V=None):
        """
        Saves mesh, geometric parameters and fiber field to an HDF5 file,
        that can be loaded for re-use with FEniCS.

        Args:
            meshfile (str): Name or relative/full path of HDF5 file to write to.
            save_fiber_vector (bool): Saves fiber vector to same hdf5 file (True) or do not save (False).
            V (dolfin vector function space): Vector function space to define the fiber vectors on (optional)
                                              This is not needed if save_fiber_vector is False,
                                              or if the fiber vectors have already been created.

            Note: if save_fiber_vector is True and V is None,
            the function object of ef should've already been created
            before calling this function.
        """
        # Write the mesh using HDF5File from FEniCS.
        with HDF5File(mpi_comm_world(), meshfile, 'w') as f:
            f.write(self.mesh(), 'mesh')

            if save_fiber_vector:
                # Write the fiber vector.
                f.write(self.fiber_vectors()[0].to_function(V), 'fiber_vector')

        # Add the geometric parameters to the same HDF5 file.
        if MPI.rank(mpi_comm_world()) == 0:
            with HDF5File(mpi_comm_self(), meshfile, 'a') as f:
                geometric_parameters = self.parameters['geometry']
                # Create (empty) geometry dataset in hdf5 file.
                f.write(np.array([], dtype=np.float_), 'geometry')
                # Add geometry parameters as attributes to the geometry dataset.
                geo_data = f.attributes('geometry')
                for k, v in geometric_parameters.items():
                    geo_data[k] = v

    @staticmethod
    def read_geometric_parameters(meshfile):
        """
        Reads geometric parameters from HDF5 file.

        Args:
            meshfile (str): Path to HDF5 file that contains the geometric parameters
                            (as saved with the function save_mesh_to_hdf5).

        Returns:
            A dolfin parameter set with the geometric parameters
            (radii of ellipsoids and truncation height).
        """
        # Create parameter set.
        geometry = Parameters('geometry')
        # Read HDF5 file.
        with HDF5File(mpi_comm_world(), meshfile, 'r') as f:
            for k, v in f.attributes('geometry').items():
                # Collect the the geometric parameters in the parameter set.
                geometry.add(k, v)
        return geometry

    def save_vectors_and_coordinate_systems(self, V, dir_out='.'):
        """
        Saves cardiac vectors, fiber vectors, wall-bounded coordinates, and fiber angles to XDMF files
        for vizualization in ParaView.
        :param V: FEniCS VectorFunctionSpace to evaluate the vector fields and coordinate systems on.
        :param dir_out: path to the output directory.
        """
        # Vectors and coordinate systems (all cvbtk.CoordinateSystem objects).
        cardiac_vectors = self.cardiac_vectors()
        fiber_vectors = self.fiber_vectors()
        wallbounded_coordinates = self.wallbounded_coordinates()
        fiber_angles = self.fiber_angles()

        mesh = V.ufl_function_space().mesh()
        V_q = FunctionSpace(mesh, VectorElement(family="Quadrature",
                                                 cell = mesh.ufl_cell(),
                                                 degree = parameters['form_compiler']['quadrature_degree'],
                                                 quad_scheme="default"))
        Q_q = vector_space_to_scalar_space(V_q)

        # The vectors can be exported to XDMF files for visualization in ParaView.
        # Cardiac vectors (note that projection not exact at nodes).
        save_to_disk(project(cardiac_vectors[0].to_function(V_q), V), os.path.join(dir_out, 'ec.xdmf'))
        save_to_disk(project(cardiac_vectors[1].to_function(V_q), V), os.path.join(dir_out, 'el.xdmf'))
        save_to_disk(project(cardiac_vectors[2].to_function(V_q), V), os.path.join(dir_out, 'et.xdmf'))

        # Fiber vectors (note that projection not exact at nodes).
        save_to_disk(project(fiber_vectors[0].to_function(V_q), V), os.path.join(dir_out, 'ef.xdmf'))
        save_to_disk(project(fiber_vectors[1].to_function(V_q), V), os.path.join(dir_out, 'es.xdmf'))
        save_to_disk(project(fiber_vectors[2].to_function(V_q), V), os.path.join(dir_out, 'en.xdmf'))

        # Normalized coordinates (wall bounded).
        Q = vector_space_to_scalar_space(V)
        save_to_disk(wallbounded_coordinates[0].to_function(Q), os.path.join(dir_out, 'u.xdmf'))
        save_to_disk(wallbounded_coordinates[1].to_function(Q), os.path.join(dir_out, 'v_wall.xdmf'))
        save_to_disk(wallbounded_coordinates[2].to_function(Q), os.path.join(dir_out, 'v_septum.xdmf'))

        # Helix and transverse angles.
        save_to_disk(project(fiber_angles[0].to_function(Q_q), Q), os.path.join(dir_out, 'ah.xdmf'))
        save_to_disk(project(fiber_angles[1].to_function(Q_q), Q), os.path.join(dir_out, 'at.xdmf'))

        # # Divergence of fiber vector.
        # ef = fiber_vectors[0].to_function(V)
        # Vef = ef.ufl_function_space()
        # Qef = vector_space_to_scalar_space(Vef)
        # div_ef = project(div(ef), Qef)
        # save_to_disk(div_ef, os.path.join(dir_out, 'div_ef.xdmf'))

        # # Gradients of x, y, z - components of fiber vectors.
        # efx, efy, efz = ef[0], ef[1], ef[2]
        # grad_efx = BasisVectorsBayer.project_gradient(efx, Vef) # This project method for the gradient is faster than project(grad(efx), V)
        # grad_efy = BasisVectorsBayer.project_gradient(efy, Vef)
        # grad_efz = BasisVectorsBayer.project_gradient(efz, Vef)
        # save_to_disk(grad_efx, os.path.join(dir_out, 'grad_efx.xdmf'))
        # save_to_disk(grad_efy, os.path.join(dir_out, 'grad_efy.xdmf'))
        # save_to_disk(grad_efz, os.path.join(dir_out, 'grad_efz.xdmf'))

        # # Interpolation factors.
        # save_to_disk(scalar_to_function(geometry._unit_vectors_bayer.t_endo(), V), os.path.join(dir_out, 't_endo_bayer.xdmf'))
        # save_to_disk(scalar_to_function(geometry._unit_vectors_bayer.t_interp(), V), os.path.join(dir_out, 't_interp_bayer.xdmf'))


# TODO This is not properly working. Tags are wrong. Volume is wrong.
class LeftVentricleVADGeometry(LeftVentricleGeometry):
    """
    Alternative version of LeftVentricleGeometry with the apex removed.

    Args:
        meshfile (optional): Path to an existing mesh to load in.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, meshfile=None, **kwargs):
        super(LeftVentricleVADGeometry, self).__init__(meshfile=meshfile,
                                                       **kwargs)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this geometry.
        """
        prm = LeftVentricleGeometry.default_parameters()
        prm.add('lvad_tube_diameter', 1.0)  # cm
        return prm

    @property
    def tube(self):
        """
        Return the surface number of the LVAD insert tube.
        """
        return 4

    def _create_mesh(self):
        """
        Helper method to interface with :class:`~cvbtk.LeftVentricleVADMesh`.
        """
        C = self.parameters['focus_height']
        h = self.parameters['truncation_height']
        e1 = self.parameters['inner_eccentricity']
        e2 = self.parameters['outer_eccentricity']
        segments = self.parameters['mesh_segments']
        resolution = self.parameters['mesh_resolution']
        r_tube = 0.5*self.parameters['lvad_tube_diameter']
        return LeftVentricleVADMesh(C, h, e1, e2, r_tube, resolution,
                                    segments=segments)

    def _create_tags(self):
        """
        Helper method to mark the basal, inner, outer, and LVAD tube insert
        surface boundaries.
        """
        # Initialize (create) the volume mesh and boundary (surface) mesh.
        mesh = self.mesh()
        b_mesh = BoundaryMesh(mesh, 'exterior')

        # Initialize the boundary mesh and create arrays of boundary faces.
        b_map = b_mesh.entity_map(2)
        b_faces = [Facet(mesh, b_map[cell.index()]) for cell in cells(b_mesh)]

        # Create an empty list of marker values to fill in.
        b_value = np.zeros(len(b_faces), dtype=int)

        # Compute the midpoint locations of each facet on the boundary.
        X = np.array([each.midpoint() for each in b_faces])

        # Compute the sigma value of each facet using the midpoint coordinates.
        C = self.parameters['focus_height']
        sig = 0.5/C*(np.sqrt(X[:, 0]**2 + X[:, 1]**2 + (X[:, 2] + C)**2)
                     + np.sqrt(X[:, 0]**2 + X[:, 1]**2 + (X[:, 2] - C)**2))

        # Fill in the marker values using sigma in relation to the mean values.
        # noinspection PyUnresolvedReferences
        b_value[sig > sig.mean()] = self.epicardium
        # noinspection PyUnresolvedReferences
        b_value[sig < sig.mean()] = self.endocardium

        # The base is where the z-normal is vertical. Must be called last.
        n_z = np.array([each.normal()[2] for each in b_faces])
        b_value[n_z == 1] = self.base

        # The tube is where the diameter == ``lvad_tube_diameter`` and the z
        # component of the surface normal is zero.
        r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        r_tube = 0.5*self.parameters['lvad_tube_diameter']
        r_bool = np.where((np.abs(r - r_tube) < 1e-2) & (np.abs(n_z) < 1e-8))
        b_value[r_bool] = self.tube

        # Create a FacetFunction and initialize to zero.
        marker = FacetFunction('size_t', mesh, value=0)

        # Fill in the FacetFunction with the marked values and return.
        for f, v in zip(b_faces, b_value):
            marker[f] = v
        return marker


class ThickWalledSphereGeometry(BaseGeometry):
    """
    High-level interface to a half-truncated, thick-walled sphere for simulation
    and post-processing.

    Args:
        meshfile (optional): Path to an existing mesh to load in.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, meshfile=None, **kwargs):
        super(ThickWalledSphereGeometry, self).__init__(meshfile=meshfile,
                                                        **kwargs)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this geometry.
        """
        prm = Parameters('thick_walled_sphere')

        prm.add('inner_radius', 1.0)
        prm.add('outer_radius', 2.0)

        prm.add('truncation_height', 0.0)  # used for compute_volume()

        prm.add('mesh_segments', 20)
        prm.add('mesh_resolution', float())

        return prm

    def compute_volume(self, surface='inner', du=None):
        """
        Approximate the volume bounded by the given surface and displacements.

        Args:
            surface (optional): Specify either ``inner`` (default) or ``outer``.
            du (optional): Displacement to move mesh by before computing volume.

        Returns:
            Approximated volume of the geometry bounded by the given surface
            and displacements.
        """
        # Displace the mesh if needed.
        if du:
            # Check if du is defined with linear elements. Interpolate if not.
            if du.ufl_element().degree() != 1:
                u = interpolate(du, VectorFunctionSpace(self.mesh(), 'P', 1))
            else:
                u = du

            # Make a copy of the mesh to displace and displace it.
            mesh = Mesh(self.mesh())
            ALE.move(mesh, u)

        # Reference mesh requested. Not displacement or copying needed.
        else:
            mesh = self.mesh()

        # Define an integral measure over the exterior surfaces.
        ds = Measure('ds', subdomain_data=self.tags())

        # Create an array of indices where facets are on the requested surface.
        surface_ = self.endocardium if surface == 'inner' else self.epicardium
        facet_array = np.where(ds.subdomain_data().array() == surface_)[0]
        # TODO Raise error if incorrect surface is requested.

        # Create an array of the facet objects themselves.
        facet_object_array = []
        for i, j in enumerate(facets(mesh)):
            if i in facet_array:
                facet_object_array.append(j)

        # Create an array of the nodal coordinates of these facets.
        x = []
        for i in facet_object_array:
            for j in vertices(i):
                x.append(j.point().array()[:])
        x = np.array(x)

        # Compute the volume as an approximation to [0, 0, 0].
        volume = 0
        d = np.array([0, 0, self.parameters['truncation_height']])
        for i in range(0, len(x), 3):
            a = x[i + 0] - d
            b = x[i + 1] - d
            c = x[i + 2] - d
            volume += np.abs(np.dot(a, np.cross(b, c))/6)

        # Gather up the volumes to a single process and return.
        return MPI.sum(mpi_comm_world(), volume)

    @property
    def base(self):
        """
        Return the surface number of the base.
        """
        return 1

    @property
    def endocardium(self):
        """
        Return the surface number of the endocardium (inner wall).
        """
        return 2

    @property
    def epicardium(self):
        """
        Return the surface number of the epicardium (outer wall).
        """
        return 3

    def _create_mesh(self):
        """
        Helper method to interface with :class:`~cvbtk.ThickWalledSphereMesh`.
        """
        r1 = self.parameters['inner_radius']
        r2 = self.parameters['outer_radius']
        segments = self.parameters['mesh_segments']
        resolution = self.parameters['mesh_resolution']
        return ThickWalledSphereMesh(r1, r2, resolution, segments=segments)

    def _create_tags(self):
        """
        Helper method to mark the basal, inner, and outer surface boundaries.
        """
        # Initialize (create) the volume mesh and boundary (surface) mesh.
        mesh = self.mesh()
        b_mesh = BoundaryMesh(mesh, 'exterior')

        # Initialize the boundary mesh and create arrays of boundary faces.
        b_map = b_mesh.entity_map(2)
        b_faces = [Facet(mesh, b_map[cell.index()]) for cell in cells(b_mesh)]

        # Create an empty list of marker values to fill in.
        b_value = np.zeros(len(b_faces), dtype=int)

        # Compute the midpoint locations of each facet on the boundary.
        X = np.array([each.midpoint() for each in b_faces])

        # Compute the radius of each facet using the midpoint coordinates.
        r = np.sqrt(X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2)

        # Fill in the marker values using radius in relation to the mean values.
        b_value[r > r.mean()] = self.epicardium
        b_value[r < r.mean()] = self.endocardium

        # The base is where the z-normal is vertical. Must be called last.
        n_z = np.array([each.normal()[2] for each in b_faces])
        b_value[n_z == 1] = self.base

        # Create a FacetFunction and initialize to zero.
        marker = FacetFunction('size_t', mesh, value=0)

        # Fill in the FacetFunction with the marked values and return.
        for f, v in zip(b_faces, b_value):
            marker[f] = v
        return marker
