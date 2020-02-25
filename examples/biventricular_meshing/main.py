"""
This script provides an example on how to create and export a biventricular
mesh, boundary markers, and volume markers.
"""

import time
import numpy as np

from dolfin import VectorElement, FunctionSpace

from cvbtk import BiventricleGeometry, print_once, info_once, save_to_disk, quadrature_function_space

# We can inspect what parameters are passable to BiventricleGeometry:
print_once('Passable parameters to BiventricleGeometry:')
info_once(BiventricleGeometry.default_parameters(), True)

# We can set the parameters using a dictionary.
geometry_inputs = {
    'f_R': 0.55 / 2,  # R_1/(Z_1 + h): Inner LV radius-to-length ratio. [-, float]
    'f_V': 0.375,  # V_lv0/V_lvw: LV cavity-to-wall-volume ratio. [-, float]
    'f_h1': 0.40,  # h/Z_1: Ratio of the truncation height above the equator with respect to endocardial apex. [-, float]
    'f_T': 0.330,  # (R_2-R_1)/(R_4-R_3): RV over LV wall thickness. [-, float]
    'f_sep': 0.7,  # R_1sep/R_1: Septal endocardial wall curvature with respect to LV endocardial curvature. [-, float] NOTE: Functions that create the boundary tags assume this value is not > 1.
    'f_VRL': 1.,  # V_rv0/V_lv0: Fraction volume between left and right ventricular cavity. [-, float]
    'V_lvw': 160.0,  # LV wall volume. [mL, float]
    'theta_A': 0.85 * np.pi,  # Lowest attachment angle between both ventricles. [Radians, float]
    'mesh_segments': 43}  # Segments argument for mshr shapes (the higher, the smoother the mesh and the closer it is to the analytical shape). [int]

# We can change the default meshing parameters by creating a dictionary and
# passing that dictionary to geometry_inputs.
mesh_generator_parameters = {
'mesh_resolution': float(43),
'feature_threshold': float(47),
'odt_optimize': True,
'lloyd_optimize': False,
'perturb_optimize': False,
'exude_optimize': False}

# Add the mesh_generator parameters to the geometry inputs.
geometry_inputs['mesh_generator'] = mesh_generator_parameters

# Create a mesh name.
mesh_name = 'biv_pluijmert_seg{}_res{}_ft{}_odt'.format(geometry_inputs['mesh_segments'],
                                                           int(mesh_generator_parameters['mesh_resolution']),
                                                           int(mesh_generator_parameters['feature_threshold']))
print_once(mesh_name)

# Create biventricular model with these inputs.
biv = BiventricleGeometry(**geometry_inputs)

# Inspecting the parameters again show that they were accepted:
info_once(biv.parameters, True)

# The mesh is not created until it is called.
t0 = time.time()
biv_mesh = biv.mesh()
print_once('Mesh created in {} s.'.format(time.time()-t0))

# The BiventricleGeometry provides the mesh(), tags() and volume_tags()
# methods to access the actual Mesh, MeshFunction (boundary marker) and
# CellFunction (volume marker) objects.
# They can be saved to an XDMF file with the save_to_disk() helper function.
save_to_disk(biv.mesh(), 'output/biventricular_mesh/{}_mesh.xdmf'.format(mesh_name))
save_to_disk(biv.tags(), 'output/biventricular_mesh/{}_boundary_tags.xdmf'.format(mesh_name))
save_to_disk(biv.volume_tags(), 'output/biventricular_mesh/{}_volume_tags.xdmf'.format(mesh_name))

# Each Geometry object should provide a compute_volume() method.
# We can check how close we are to the requested volume with it:
print_once('Computing volumes...')
t0 = time.time()
V_1 = biv.compute_volume('lv', volume='cavity')
print_once('LV cavity computed:',V_1)
t1 = time.time()
print_once('LV cavity input:',geometry_inputs['V_lvw']*geometry_inputs['f_V'])
print_once('RV cavity computed:',biv.compute_volume('rv', volume='cavity'))
t2 = time.time()
print_once('RV cavity input:',geometry_inputs['V_lvw']*geometry_inputs['f_V'] * geometry_inputs['f_VRL'])
print_once('LV wall computed:',biv.compute_volume('lv', volume='wall'))
t3 = time.time()
print_once('LV wall input:',geometry_inputs['V_lvw'])
print_once('RV wall computed:',biv.compute_volume('rv', volume='wall'))
t4 = time.time()
print_once('Time LV cavity:', t1-t0)
print_once('Time RV cavity:', t2-t1)
print_once('Time LV wall:', t3-t2)
print_once('Time RV wall', t4-t3)

# The base, endocardium, and epicardium surfces are already marked for use.
# They correspond with the integer values in the MeshFunction object and
# can be accessed with their corresponding property:
print_once('The base is marked with: {}.'.format(biv.base))
print_once('The LV endocardium is marked with: {}.'.format(biv.lv_endocardium))
print_once('The LV epicardium is marked with: {}.'.format(biv.lv_epicardium))
print_once('The RV endocardium (free wall) is marked with: {}.'.format(biv.rv_endocardium))
print_once('The RV endocardium (septal wall) is marked with: {}.'.format(biv.rv_septum))
print_once('The RV epicardium is marked with: {}.'.format(biv.rv_epicardium))

# The (volume) cells of the mesh are also marked as either LV or RV:
print_once('The LV is marked with: {}.'.format(biv.lv_tag))
print_once('The RV is marked with: {}.'.format(biv.rv_tag))

# Note that the XDMF file format is only suitable for visualization.
# For reuse, the HDF5 file format should be used. To be able to
# recreate the markers, the geometric parameters should be saved as well.
# Both the mesh and the geometric parameters can be saved in 1 HDF5 file.
# Additionally, the fiber vectors can be saved in the same HDF5 file.

# Specify whether to compute and save the fiber vectors.
save_fiber_vector = True
if save_fiber_vector:
    # Discretize the fiber vectors onto quadrature elements.
    mesh = biv.mesh()
    quadrature_degree = 4
    V_q = quadrature_function_space(mesh, degree=quadrature_degree)
else:
    V_q = None

# We can save the mesh, geometry parameters and the fiber vectors to a HDF5 file in one command:
biv.save_mesh_to_hdf5('output/biventricular_mesh/{}_mesh.hdf5'.format(mesh_name),
                      save_fiber_vector=save_fiber_vector, V=V_q)

# Now we can reload the mesh when creating a new geometry object with use
# of the meshfile argument of BiventricleGeometry.
# We do not need to specify any inputs about the geometry or meshing,
# as those are both loaded.
biv_loaded = BiventricleGeometry(meshfile='output/biventricular_mesh/{}_mesh.hdf5'.format(mesh_name))
V_1_loaded = biv_loaded.compute_volume('lv', volume='cavity')
Ve = abs(V_1 - V_1_loaded)/V_1
print_once('The original and loaded meshes have a {}% LV volume error.'.format(Ve))
