# -*- coding: utf-8 -*-
"""
This module provides utility functions.
"""
from dolfin import (FunctionSpace, TensorFunctionSpace, VectorFunctionSpace,
                    project, conditional, lt, eq, gt, atan, ge, pi, info, VectorElement,
                    FiniteElement, parameters)
from dolfin.cpp.common import MPI, mpi_comm_world
from dolfin.cpp.io import XDMFFile
from dolfin.cpp.la import VectorSpaceBasis

import csv
import os
import matplotlib.pyplot as plt

__all__ = [
    'build_nullspace',
    'import_h5',
    'safe_project',
    'vector_space_to_scalar_space',
    'scalar_space_to_vector_space',
    'vector_space_to_tensor_space',
    'save_dict_to_csv',
    'read_dict_from_csv',
    'save_to_disk',
    'print_once',
    'info_once',
    'atan2_',
    'reset_values',
    'global_function_average',
    'figure_make_up',
    'quadrature_function_space'
]


def build_nullspace(u, modes):
    """
    Create a nullspace to restrict rigid body motion on **modes** when using an
    iterative solver.

    The **modes** argument can be any combination of the following:
        * Translation: **x**, **y**, **z**
        * Rotation: **xy**, **xz**, **yz**

    Furthermore, for the rotation modes, **xy == yx**, and so on.

    Args:
        u: The displacement unknown.
        modes: List of modes to restrict, e.g.: ['x', 'y', 'xy'].

    Returns:
        The orthonormalized :class:`VectorSpaceBasis` nullspace.
    """
    V = u.ufl_function_space()
    nullspace_vectors = []

    if 'x' in modes:
        vector = u.vector().copy()
        V.sub(0).dofmap().set(vector, 1.0)
        nullspace_vectors.append(vector)

    if 'y' in modes:
        vector = u.vector().copy()
        V.sub(1).dofmap().set(vector, 1.0)
        nullspace_vectors.append(vector)

    if 'z' in modes:
        vector = u.vector().copy()
        V.sub(2).dofmap().set(vector, 1.0)
        nullspace_vectors.append(vector)

    if 'xy' in modes or 'yx' in modes:
        vector = u.vector().copy()
        V.sub(0).set_x(vector, -1.0, 1)
        V.sub(1).set_x(vector, +1.0, 0)
        nullspace_vectors.append(vector)

    if 'xz' in modes or 'zx' in modes:
        vector = u.vector().copy()
        V.sub(0).set_x(vector, +1.0, 2)
        V.sub(2).set_x(vector, -1.0, 0)
        nullspace_vectors.append(vector)

    if 'yz' in modes or 'zy' in modes:
        vector = u.vector().copy()
        V.sub(1).set_x(vector, -1.0, 2)
        V.sub(2).set_x(vector, +1.0, 1)
        nullspace_vectors.append(vector)

    # We need to call the apply() method whenever a vector is manipulated.
    for nullspace_vector in nullspace_vectors:
        nullspace_vector.apply('insert')

    # Create the VectorSpaceBasis, orthonormalize, and return.
    nullspace = VectorSpaceBasis(nullspace_vectors)
    nullspace.orthonormalize()
    return nullspace


def import_h5(filename, varnames):
    """
    Import list-like arrays to file using :mod:`h5py`.

    Args:
        filename (str): Filename to import from.
        varnames (str or strs): Name(s) of variable(s) to import.

    Return:
        vars (list or lists): Imported variable(s).
        attr (list of tuples): Attribute(s) of the imported variable(s).
    """
    import h5py
    dout = []
    attr = []
    with h5py.File(filename, 'r') as f:
        for v in varnames:
            dout += [f[v][...]]
            att_buf = []
            for att in f[v].attrs.items():
                att_buf += [att]
            attr += [att_buf]
    return dout, attr


def safe_project(u, V):
    """
    Project **u** onto **V**.

    If **V** is not a linear function space, then **u** is projected to a linear
    function space first before being interpolated to **V**.

    Args:
        u: :class:`~dolfin.Function`
        V: :class:`~dolfin.FunctionSpace`

    Returns:
        **u** projected onto **V**.
    """
    # if V.ufl_element().degree() != 1:
    #     mesh, family = V.ufl_domain().ufl_cargo(), V.ufl_element().family()
    #
    #     if V.ufl_element().value_size() == 3:
    #         V1 = VectorFunctionSpace(mesh, family, 1)
    #
    #     elif V.ufl_element().value_size() == 9:
    #         V1 = TensorFunctionSpace(mesh, family, 1)
    #
    #     else:
    #         V1 = FunctionSpace(mesh, family, 1)
    #
    #     return interpolate(project(u, V1), V)
    #
    # else:
    #     return project(u, V)
    # TODO Maybe deprecate this function. Use project from now on?
    return project(u, V)


def vector_space_to_scalar_space(V):
    """
    Returns a scalar function space defined on the same mesh with the same
    elements as the given vector function space.

    Args:
        V: :class:`~dolfin.VectorFunctionSpace`

    Returns:
        :class:`~dolfin.FunctionSpace`
    """
    # Extract the mesh from the given vector function space.
    mesh = V.ufl_domain().ufl_cargo()

    # Extract the element parameters from the given vector function space.
    family = V.ufl_element().family()
    degree = V.ufl_element().degree()

    if family == 'Quadrature':
        # Create a tensor function space from a finite element.
        cell = V.ufl_element().cell()
        quad_scheme = V.ufl_element().quadrature_scheme()
        element = FiniteElement(family=family,
                                cell=cell,
                                degree=degree,
                                quad_scheme=quad_scheme)
        return FunctionSpace(mesh, element)
    else:
        # Create and return a scalar function space.
        return FunctionSpace(mesh, family, degree)

def scalar_space_to_vector_space(Q):
    """
    Returns a vector function space defined on the same mesh with the same
    elements as the given scalar function space.

    Args:
        Q: :class:`~dolfin.FunctionSpace`

    Returns:
        :class:`~dolfin.VectorFunctionSpace`
    """
    # Extract the mesh from the given vector function space.
    mesh = Q.ufl_domain().ufl_cargo()

    # Extract the element parameters from the given vector function space.
    family = Q.ufl_element().family()
    degree = Q.ufl_element().degree()

    if family == 'Quadrature':
        # Create a tensor function space from a vector element.
        cell = Q.ufl_element().cell()
        quad_scheme = Q.ufl_element().quadrature_scheme()
        element = VectorElement(family=family,
                                cell=cell,
                                degree=degree,
                                quad_scheme=quad_scheme)
        return FunctionSpace(mesh, element)
    else:
        # Create and return a scalar function space.
        return VectorFunctionSpace(mesh, family, degree)


def vector_space_to_tensor_space(V):
    """
    Returns a tensor function space defined on the same mesh with the same
    elements as the given vector function space.

    Args:
        V: :class:`~dolfin.VectorFunctionSpace`

    Returns:
        :class:`~dolfin.TensorFunctionSpace`
    """
    # Extract the mesh from the given vector function space.
    mesh = V.ufl_domain().ufl_cargo()

    # Extract the element parameters from the given vector function space.
    family = V.ufl_element().family()
    degree = V.ufl_element().degree()

    if family == 'Quadrature':
        raise NotImplementedError
    else:
        # Create and return a tensor function space.
        return TensorFunctionSpace(mesh, family, degree)

def save_dict_to_csv(d, filename):
    """
    Writes a (nested) dictionary to a csv file.

    Args:
         d (dictionary): Dcitionary to save
         filename (str): Filename for the csv file (including .csv)
    """
    # Check if output directory exists
    if MPI.rank(mpi_comm_world()) == 0:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    def flatten_dict(d, path='', d_out=None):

        if d_out is None:
            d_out = {}

        for key in d.keys():
            path_new = '{}/{}'.format(path, key)
            if type(d[key]) is dict:
                d_out = flatten_dict(d[key], path=path_new, d_out=d_out)
            else:
                d_out.update({path_new: d[key]})

        return d_out

    def write_dict(d, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            for key, value in sorted(d.items()):
                writer.writerow([key, value])

    write_dict(flatten_dict(d), filename)

def read_dict_from_csv(filename):
    """
    Reads a (nested) dictionary from a csv file (reverse of save_dict_to_csv).

    Args:
         filename (str): Filename of the csv file (including .csv)

         Returns:
             Nested dictionary based on the CSV file
    """

    # Check if the file exists.
    if not os.path.exists(filename):
        raise FileNotFoundError('File "{}" does not exist.'
                                .format(filename))

    def nested_dict(d, keys, value):
        # Set d['a']['b']['c'] = value, when keys = ['a', 'b', 'c']
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    d_out={}

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            path = line[0]

            # Convert to ints or floats if possible (otherwise, keep it as a string).
            try:
                if '.' in line[1] or 'e' in line[1]:
                    value = float(line[1])
                else:
                    value = int(line[1])
            except ValueError:
                if line[1] == 'False':
                    value = False
                elif line[1] == 'True':
                    value = True
                else:
                    value = line[1]

            if path[0]=='/':
                # Exclude first slash as it would create an empty entry when applying path.split('/')
                path = path[1:]

            nested_dict(d_out, path.split('/'), value)

    return d_out

def save_to_disk(o, filename, t=None):
    """
    Helper function to write a FEniCS Mesh, MeshFunction, or Function object to
    an XDMF file format for manual inspection in ParaView.

    Args:
        o: Object to write.
        filename: Name or relative/full path of XDMF file to write to.
        t (optional): timestamp.
    """
    if t is None:
        with XDMFFile(filename) as f:
            f.write(o)
    else:
        with XDMFFile(filename) as f:
            f.write(o, t)

def print_once(*message):
    """
    Help function to print a message when running in parallel.
    It prints the message only once, using the node with rank 0.

    Args:
        *message (str): String(s) with message(s) to print.
    """
    if MPI.rank(mpi_comm_world()) == 0:
        print(*message)

def info_once(object, verbose=False):
    """
    Help function to print a message when running in parallel.
    It prints the message only once, using the node with rank 0.

    Args:
        Same as to dolfin's info function.
    """
    if MPI.rank(mpi_comm_world()) == 0:
        info(object, verbose)

def atan2_(y, x):
    """
    Custom UFl function for inverse tangent 2.

    https://en.wikipedia.org/wiki/Atan2

    Note: does not check for x==0 and y==0. Returns 0 instead.

    :param y: Nominator.
    :param x: Denominator.
    :return: UFL object.
    """
    x_lt_z = conditional(lt(x, 0), 1, 0)
    x_eq_z = conditional(eq(x, 0), 1, 0)
    y_lt_z = conditional(lt(y, 0), 1, 0)

    cd_1 = conditional(gt(x, 0), 1, 0)
    f_1 = atan(y/x)

    cd_2 = x_lt_z * conditional(ge(y, 0), 1, 0)
    f_2 = f_1 + pi

    cd_3 = x_lt_z * y_lt_z
    f_3 = f_1 - pi

    cd_4 = x_eq_z * conditional(gt(y, 0), 1, 0)
    f_4 = 0.5*pi

    cd_5 = x_eq_z * y_lt_z
    f_5 = -0.5*pi

    return cd_1*f_1 + cd_2*f_2 + cd_3*f_3 + cd_4*f_4 + cd_5*f_5


def reset_values(function_to_reset, array_to_reset_from):
    """
    Helper function to reset DOLFIN quantities.
    """
    function_to_reset.vector()[:] = array_to_reset_from
    function_to_reset.vector().apply('')


def global_function_average(func):
    """
    Helper function to compute the global average of the dof values
    of a dolfin function across all processes.

    Args:
        func (dolfin.Function): Function.

    Returns:
        avg (float): global average of dof values of func.
    """
    array = func.vector().array()

    sum_local = sum(array)
    n_local = len(array)

    sum_global = MPI.sum(mpi_comm_world(), sum_local)
    n_global = MPI.sum(mpi_comm_world(), n_local)

    return sum_global / n_global


def figure_make_up(title=None, xlabel=None, ylabel=None,
                   create_legend=True, legend_title=None, fontsize=None):
    if title is not None:
        plt.title(title, fontsize=fontsize * 1.15)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize * 0.9)
    if create_legend is True:
        leg = plt.legend(title=legend_title, fontsize=fontsize * 0.9)
        leg.get_title().set_fontsize(fontsize * 0.9)


def quadrature_function_space(mesh, degree=None):
    """
    Helper-function to define a quadrature function space.

    Args:
        mesh: a dolfin Mesh.
        degree: quadrature degree. If not specified, the global quadrature degree
                is used if it is set. If it is not set, an error will be raised.
    """
    if degree is None:
        # Read global quadrature degree
        degree = parameters['form_compiler']['quadrature_degree']

        # Check if quadrature degree is set in parameters.
        if degree is None:
            # Degree was not set in parameters['form_compiler']['quadrature_degree'].
            raise RuntimeError(
                "Quadrature degree not specified in parameters['form_compiler']['quadrature_degree']")

    # Create quadrature element and function space.
    quad_element_V = VectorElement(family="Quadrature",
                                   cell=mesh.ufl_cell(),
                                   degree=degree,
                                   quad_scheme="default")

    return FunctionSpace(mesh, quad_element_V)



# def configure_logging(filename):
#     try:
#         with open(filename, 'r') as f:
#             logging.config.dictConfig(json.load(f))
#
#     except FileNotFoundError:
#         logging.basicConfig()
#         logging.getLogger().setLevel('DEBUG')
#
#     # Set FEniCS loggers to WARNING.
#     logging.getLogger('FFC').setLevel(logging.WARNING)
#     logging.getLogger('UFL').setLevel(logging.WARNING)
#
#
# def inputs_from_json(filename):
#     log = logging.getLogger(__name__)
#
#     try:
#         with open(filename, 'r') as f:
#             inputs = json.load(f)
#
#     except FileNotFoundError:
#             if MPI.rank(mpi_comm_world()) == 0:
#                 log.error('%s does not exist', filename)
#             raise
#
#     else:
#         if MPI.rank(mpi_comm_world()) == 0:
#             log.info('Loaded %s into a dictionary', filename)
#
#     return inputs
#
#
# def split_vector_functions(vector_function, function_space):
#     # Create a function assigner from V --> Q.
#     Q = function_space
#     V = vector_function.ufl_function_space()
#     fa = [FunctionAssigner(Q, V.sub(i)) for i in range(3)]
#
#     # Split and assign from V --> Q.
#     u = vector_function
#     u_x, u_y, u_z = [Function(Q) for _ in range(3)]
#     [fa[i].assign(j, u.split()[i]) for i, j in enumerate([u_x, u_y, u_z])]
#     [each.vector().apply('') for each in [u_x, u_y, u_z]]
#
#     return u_x, u_y, u_z
#
#
# def parse_parameters(prm, **kwargs):
#     for k, v in prm.items():
#         value = kwargs.get(k)
#         if value is not None:
#             if isinstance(v, bool):
#                 prm[k] = bool(value)
#             elif isinstance(v, str):
#                 prm[k] = str(value)
#             elif isinstance(v, int):
#                 prm[k] = int(value)
#             elif isinstance(v, float):
#                 prm[k] = float(value)
#             else:
#                 prm[k].update(value)

# def export_h5(filename, varnames, vars, attrs=None):
#     """Export list-like arrays to file using :mod:`h5py`.
#
#     Credit to Carlos Ledezma <c.ledezma@ucl.ac.uk> for providing me with this.
#
#     Args:
#         filename (str): Filename to export to.
#         varnames (str or strs): Name(s) of variable(s) to export.
#         vars (list or lists): Variable(s) to export.
#         attrs (list or lists): Attribute(s) corresponding to variables, given
#             for each variable as [[name1, value1], [name2, value2], ...].
#     """
#     import h5py
#     with h5py.File(filename, 'w') as f:
#         for i in range(len(varnames)):
#             dset = f.create_dataset(varnames[i], vars[i].shape, vars[i].dtype)
#             dset[...] = vars[i]
#             if attrs is not None:
#                 for attr in attrs[i]:
#                     dset.attrs.create(attr[0], attr[1])
