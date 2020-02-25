"""
This script demonstrates the use of quadrature elements, which can be used to define fiber vectors
on the quadrature points, which prevents the need for interpolation between Lagrange nodes to find
the fiber vector at quadrature points.
"""
import warnings

from dolfin import FunctionSpace, VectorFunctionSpace, Function, interpolate, Expression, grad, project, \
    VectorElement, FiniteElement, sqrt, cross, parameters, as_matrix, as_tensor, dot, TensorElement
from dolfin.cpp.common import mpi_comm_world
from dolfin.cpp.io import HDF5File
from dolfin.cpp.mesh import UnitCubeMesh

from cvbtk import save_to_disk, reset_values, fiber_stretch_ratio, vector_space_to_tensor_space, \
    right_cauchy_green_deformation, vector_space_to_scalar_space, CoordinateSystem
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')


def norm(v):
    """
    Returns norm of vector v (as UFL expression).
    """
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize(v):
    """
    Returns normalized vector v (as UFL expression).
    """
    return v/norm(v)


# Specify output directory.
dir_out = 'output/quadrature_elements/'

# SPecify global quadrature degree.
parameters['form_compiler']['quadrature_degree'] = 4

# Create simple cube mesh.
mesh = UnitCubeMesh(4,4,4)

# Create quadrature element for vectors (dofs are at quadrature points).
quad_element_V = VectorElement(family = "Quadrature",
                             cell = mesh.ufl_cell(),
                             degree = parameters['form_compiler']['quadrature_degree'],
                             quad_scheme="default")

# Create quadrature element for scalars (dofs are at quadrature points).
quad_element_Q = FiniteElement(family = "Quadrature",
                             cell = mesh.ufl_cell(),
                             degree = parameters['form_compiler']['quadrature_degree'],
                             quad_scheme="default")

# FunctionSpaces with quadrature elements.
V_q = FunctionSpace(mesh, quad_element_V)
Q_q = FunctionSpace(mesh, quad_element_Q)

# Create normal Lagrange function spaces.
Q = FunctionSpace(mesh, "Lagrange", 2)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Create a scalar Function u.
u = interpolate(Expression("pow(x[0],2)", degree = 3), Q)

# Compute grad(u) at the quadrature points:
# The following would raise an error:
# ef_q = project(grad(u), V_q)

# We cannot project expressions on a quadrature element space, but we can interpolate functions onto it:
# First project grad(u) on a Lagrange space.
ef = project(grad(u), V)  # We use the name ef here, but note its an arbitrary (non-unit) vector.
# Secondly, Interpolate the projected grad(u) onto the quadrature function space.
ef_q = interpolate(ef, V_q)

# To verify what interpolate does, we check whether it is the same as the following:
X_lagrange = Q.tabulate_dof_coordinates().reshape(-1, 3)  # Coordinates of Lagrangian nodes.
X_quad = Q_q.tabulate_dof_coordinates().reshape(-1, 3)  # Coordinates of quadrature points.
error = []
quad_dofs = ef_q.vector().array().reshape(-1, 3)  # ef vector at quadrature points.
for ii in range(len(quad_dofs)):
    q_dof = quad_dofs[ii, :]  # ef at current quadrature point (created by using the interpolate function above).
    x, y, z = X_quad[ii, :]  # coordinates of current quadrature point.
    ef_int = ef(x, y, z)  # Compute ef at point (x, y, z) using this syntax (it's doing interpolation between nodes).
    error.append(sum(abs(q_dof - ef_int)))

# We find errors of machine precision magnitude, so now know what interpolate does.
print('sum(error) of interpolated function: {} \nmean(error) : {}'.format(sum(error), np.mean(error)))

# Note that quadrature elements need more memory than Lagrange elements.
print('Number of nodes in quadrature function space:', X_quad.shape[0])
print('Number of nodes in Lagrange function space:', X_lagrange.shape[0])

# To visualize the vectors in the quadrature element we have to project it on a suitable function space (e.g. Lagrange):
ef_q_projected = project(ef_q, V)
save_to_disk(ef_q_projected, dir_out+'ef_q.xdmf')

# We can do computations with the quadrature element and project the solutions on a Lagrange function space.
# Rotation matrix 90 degrees z-axis.
Qr = as_matrix([[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]])

# Rotate grad(u) 90 degrees.
u_r = project(Qr*grad(u), V)

# Compute the cross product of grad(u) (defined on quadrature function space) and the 90 degree rotated grad(u).
cross_ = project(cross(Qr*grad(u), ef_q), V)

# Visualize the results: we find three orthogonal vectors (ef, u_r, cross)
save_to_disk(ef, dir_out+'ef.xdmf')
save_to_disk(u_r, dir_out+'u_r.xdmf')
save_to_disk(cross_, dir_out+'cross.xdmf')
save_to_disk(project(ef_q - ef, V), dir_out+'ef_q_min_ef.xdmf')

# Normalize the values of the quadrature function manually.
ef_q_n = Function(V_q)
ef_q_n_array = []
for ii in range(len(quad_dofs)):
    q_dof = quad_dofs[ii, :]
    q_dof /= np.linalg.norm(q_dof)
    ef_q_n_array.append(q_dof)
ef_q_n_array = np.array(ef_q_n_array)
reset_values(ef_q_n, ef_q_n_array.reshape(-1))

# Visualize the result.
save_to_disk(project(ef_q_n, V), dir_out+'ef_q_n.xdmf')

# We could also get this results by expressing the normalization as a UFL expression:
ef_q_n_projected = project(normalize(ef_q), V)
save_to_disk(ef_q_n_projected, dir_out+'normalize_ef_q.xdmf')

# Due to projection, we will find non-unit magnitudes at nodes of the Lagrange function space:
save_to_disk(project(norm(ef_q_n_projected), Q), dir_out+'norm_ef_q_n_projected.xdmf')

# If we project the norm of the normalized vectors in the quadrature function space, we do find the correct result.
save_to_disk(project(norm(ef_q_n), Q), dir_out+'norm_ef_q_n.xdmf')

# We can use the vectors in the quadrature element to compute e.g. fiber stretch:
fs = fiber_stretch_ratio(u_r, ef_q_n)
save_to_disk(project(fs, Q), dir_out+'fiber_stretch.xdmf')

# Or as_tensor (create random tensor and results from using only quadrature elements):
fsn = as_tensor((ef_q_n, Qr*ef_q_n, ef_q_n))
save_to_disk(project(sqrt(dot(ef_q, fsn*ef_q)), Q), dir_out+'as_tensor.xdmf')

# Save the quadrature function:
with HDF5File(mpi_comm_world(), dir_out+'meshfile.hdf5', 'w') as f:
    f.write(ef_q_n, 'fiber_vector')

# Load the quadrature function:
with HDF5File(mpi_comm_world(), dir_out+'meshfile.hdf5', 'r') as f:
    # Retrieve element signature
    attr = f.attributes('fiber_vector')
    element = attr['signature']
    family = element.split(',')[0].split('(')[-1][1:-1]
    cell =  element.split(',')[1].strip()
    degree = int(element.split(',')[2])
    quad_scheme =  element.split(',')[3].split('=')[1].split(')')[0][1:-1]

    # Check if the loaded quadrature degree corresponds to te current quadrature degree.
    if degree != parameters['form_compiler']['quadrature_degree']:
        warnings.warn("The quadrature degree of the loaded fiber vectors is not the same as the current quadrature degree in parameters['form_compiler']['quadrature_degree']")

    # Create function space.
    element_V = VectorElement(family=family,
                              cell=cell,
                              degree=degree,
                              quad_scheme=quad_scheme)
    V = FunctionSpace(mesh, element_V)

    ef_q_n_loaded = Function(V)
    f.read(ef_q_n_loaded, 'fiber_vector')

print('Reloading error:',sum(abs(ef_q_n.vector().array() - ef_q_n_loaded.vector().array())))