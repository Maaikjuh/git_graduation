"""
This script demonstrates how the cylindrical vectors are used and
exports XDMF files of them for visualization in ParaView.
"""
import cvbtk.resources
from dolfin import parameters, VectorFunctionSpace, project, VectorElement, FunctionSpace, acos, inner
import os

from cvbtk import save_to_disk, CoordinateSystem, CylindricalBasisVectors, vector_space_to_scalar_space

# Set output directory.
dir_out = 'output/cylindrical_vectors'

# Set quadrature rule.
parameters['form_compiler']['quadrature_degree'] = 4

# Load geometry.
geometry = cvbtk.resources.reference_biventricle()

# Define a Function space.
V = VectorFunctionSpace(geometry.mesh(), 'Lagrange', 1)
Q = vector_space_to_scalar_space(V)

# Quadrature Function space.
mesh = geometry.mesh()

# Create quadrature element and function space.
quad_element_V = VectorElement(family="Quadrature",
                               cell=mesh.ufl_cell(),
                               degree=parameters['form_compiler']['quadrature_degree'],
                               quad_scheme="default")
V_q = FunctionSpace(mesh, quad_element_V)

# Extract cylindrical vectors.
ez, er, ec = geometry.cylindrical_vectors()

# LV free wall analytical cylindrical vectors.
ez_lv = CoordinateSystem(CylindricalBasisVectors.ez)
er_lv = CoordinateSystem(CylindricalBasisVectors.er)
ec_lv = CoordinateSystem(CylindricalBasisVectors.ec, ez_lv, er_lv)

# Save LV free wall analytical vectors (project since they are defined on quad space.
save_to_disk(project(ez_lv.to_function(V_q), V), os.path.join(dir_out, 'ez_lv.xdmf'))
save_to_disk(project(er_lv.to_function(V_q), V), os.path.join(dir_out, 'er_lv.xdmf'))
save_to_disk(project(ec_lv.to_function(V_q), V), os.path.join(dir_out, 'ec_lv.xdmf'))

try:
    # If vectros are CoordinateSystem object, we can save the functions like this.
    save_to_disk(ez.to_function(V), os.path.join(dir_out, 'ez.xdmf'))
    save_to_disk(er.to_function(V), os.path.join(dir_out, 'er.xdmf'))
    save_to_disk(ec.to_function(V), os.path.join(dir_out, 'ec.xdmf'))

except AttributeError:
    # Project (vectors are UFL-like objects) and save them.
    save_to_disk(project(ez, V), os.path.join(dir_out, 'ez.xdmf'))
    save_to_disk(project(er, V), os.path.join(dir_out, 'er.xdmf'))
    save_to_disk(project(ec, V), os.path.join(dir_out, 'ec.xdmf'))

    # Project (vectors are UFL-like objects) and save them.
    save_to_disk(project(acos(inner(ez, ez_lv.to_function(V_q)))*180/3.1415, Q), os.path.join(dir_out, 'diff_ez.xdmf'))
    save_to_disk(project(acos(inner(er, er_lv.to_function(V_q)))*180/3.1415, Q), os.path.join(dir_out, 'diff_er.xdmf'))
    save_to_disk(project(acos(inner(ec, ec_lv.to_function(V_q)))*180/3.1415, Q), os.path.join(dir_out, 'diff_ec.xdmf'))
