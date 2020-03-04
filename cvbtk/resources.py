# -*- coding: utf-8 -*-
"""
This module provides functions that return reference data, meshes, and so on,
for use in testing and comparisons.
"""
import pkg_resources

from cvbtk.utils import print_once
from .dataset import Dataset
from .geometries import LeftVentricleGeometry, BiventricleGeometry

__all__ = ['reference_hemodynamics',
           'reference_left_ventricle',
           'reference_left_ventricle_pluijmert',
           'reference_biventricle',
           
           #added by Maaike
           'select_left_ventricle_mesh']

DATA_DIRECTORY = 'data/'


def reference_hemodynamics(dataset='bovendeerd2009_sepran'):
    """
    Returns a Dataset with reference data to use for testing and comparison.

    Args:
        dataset: Reference dataset to use. Defaults to 'bovendeerd2009_sepran'.
    """
    datafile = ''.join([DATA_DIRECTORY, 'hemodynamics_', dataset, '.csv'])
    return Dataset(filename=pkg_resources.resource_filename(__name__, datafile))


def reference_left_ventricle(resolution=30, **kwargs):
    """
    Returns a pre-defined :class:`~cvbtk.LeftVentricleGeometry` for testing and
    comparison.

    The fiber field parameters will need to be defined before creation of the
    fiber field basis vectors, otherwise they will use the default values.

    Args:
        resolution: Pre-defined resolution to use. Choose from 30(*), 40, or 50.
        **kwargs: optional keyword arguments to add to the input for the geometry.
    """
    inputs = {'wall_volume': 136.0,
              'cavity_volume': 44.0,
              'focus_height': 4.3,
              'truncation_height': 2.4,
              'mesh_segments': int(resolution),
              'mesh_resolution': float(resolution)
              }

    inputs.update(kwargs)

    # TODO Check that the resolution number is valid.
    meshfile = DATA_DIRECTORY + 'mesh_leftventricle_{}.hdf5'
    meshfile = meshfile.format(int(resolution))
    datafile = pkg_resources.resource_filename(__name__, meshfile)
    return LeftVentricleGeometry(meshfile=datafile, **inputs)


def reference_left_ventricle_pluijmert(resolution=30, **kwargs):
    """
    Returns a pre-defined :class:`~cvbtk.LeftVentricleGeometry` for testing and
    comparison.

    The fiber field parameters will need to be defined before creation of the
    fiber field basis vectors, otherwise they will use the default values.

    Size of the LV is similar to the LV part of the BiV mesh presented by Pluijmert et al. (2017)
    in terms of cavity and wall volumes, focus height, and truncation height.

    Args:
        resolution: Pre-defined resolution to use. Choose from 30(*).
        **kwargs: optional keyword arguments to add to the input for the geometry.
    """
    inputs = {'wall_volume': 160.0,
              'cavity_volume': 60.0,
              'focus_height': 4.85,
              'truncation_height': 2.1015,
              'mesh_resolution': float(resolution),
              'mesh_segments': 20}

    inputs.update(kwargs)

    # TODO Check that the resolution number is valid.
    meshfile = DATA_DIRECTORY + 'mesh_leftventricle_pluijmert_{}.hdf5'
    meshfile = meshfile.format(int(resolution))
    datafile = pkg_resources.resource_filename(__name__, meshfile)
    return LeftVentricleGeometry(meshfile=datafile, **inputs)

def select_left_ventricle_mesh(filename='lv_maaike_seg30_res30_mesh', resolution=30, **kwargs):
    meshfile = DATA_DIRECTORY + filename
    datafile = pkg_resources.resource_filename(__name__, meshfile)
    return LeftVentricleGeometry(meshfile=datafile, **kwargs)


def reference_biventricle(resolution=43, **kwargs):
    """
    Returns a pre-defined :class:`~cvbtk.BiventricleGeometry` for testing and
    comparison.

    The fiber field parameters will need to be defined before creation of the
    fiber field basis vectors, otherwise they will use the default values.

    Shape is identical to the reference BiV mesh presented by Pluijmert et al. (2017).

    Args:
        resolution: Pre-defined resolution to use. Choose from 43(*).
        **kwargs: Optional parameters for the Geometry. Note that an existing mesh
                         will be loaded, so some geometry inputs will be loaded, overwriting inputs passed on here.
                         Typical parameters to pass are parameters for the Bayer vectorization (geometry_inputs['bayer'])
                         and the fiber field (geometry_inputs['fiber_field']).
    """
    # Check that the resolution number is valid.
    if not resolution in [43]:
        raise ValueError('Unavailable resolution.')

    # TODO:
    # We have 2 BiV meshed with reoriented fibers. Currently, the one I used in my thesis
    # (mesh_biventricle_43_REF_reoriented.hdf5) is loaded. However, one may prefer the
    # other fiber field in mesh_biventricle_43_ADAPTED_BAYER_reoriented.hdf5
    meshfile = DATA_DIRECTORY + 'mesh_biventricle_{}_REF_reoriented.hdf5'
    meshfile = meshfile.format(int(resolution))
    datafile = pkg_resources.resource_filename(__name__, meshfile)
    return BiventricleGeometry(meshfile=datafile, **kwargs)
