"""
This script provides an example on how to create and export a left ventricular mesh.

"""

import time
import numpy as np
from dolfin.cpp.common import mpi_comm_world
from dolfin.cpp.mesh import Mesh
from dolfin import VectorElement, FunctionSpace
from dolfin.cpp.io import HDF5File, XDMFFile
from cvbtk import LeftVentricleGeometry, print_once, info_once, save_to_disk, quadrature_function_space

# We can inspect what parameters are passable to LeftVentricleGeometry:
print_once('Passable parameters to LeftVentricleGeometry:')
info_once(LeftVentricleGeometry.default_parameters(), True)

geometry_inputs = {
 #   'cavity_volume': 44.,  
    'focus_height': 4.3,  
    'mesh_resolution': 30.,
    'inner_eccentricity': 0.934819,   
    'mesh_segments': 30,  
    'outer_eccentricity': 0.807075,  
    'truncation_height': 2.4}   
#    'wall_volume': 136.,
 #   'load_fiber_field_from_meshfile':False} 

# We can change the default meshing parameters by creating a dictionary and
# passing that dictionary to geometry_inputs.
#mesh_generator_parameters = {
#'mesh_resolution': float(20),
#'odt_optimize': True,
#'lloyd_optimize': False,
#'perturb_optimize': False,
#'exude_optimize': False}

# Add the mesh_generator parameters to the geometry inputs.
#geometry_inputs['mesh_generator'] = mesh_generator_parameters

# Create a mesh name.
mesh_name = 'lv_maaike_seg{}_res{}'.format(geometry_inputs['mesh_segments'],
                                                           int(geometry_inputs['mesh_resolution']),)
print_once(mesh_name)

# Create leftventricular model with these inputs.
lv = LeftVentricleGeometry(**geometry_inputs)

# Inspecting the parameters again show that they were accepted:
info_once(lv.parameters, True)

# The mesh is not created until it is called.
t0 = time.time()
lv_mesh = lv.mesh()
print_once('Mesh created in {} s.'.format(time.time()-t0))

# Specify whether to compute and save the fiber vectors.
save_fiber_vector = None
if save_fiber_vector:
    # Discretize the fiber vectors onto quadrature elements.
    mesh = lv.mesh()
    quadrature_degree = 4
    V_q = quadrature_function_space(mesh, degree=quadrature_degree)
else:
    V_q = None

# We can save the mesh, geometry parameters and the fiber vectors to a HDF5 file in one command:
lv.save_mesh_to_hdf5('output/leftventricular_mesh/{}_mesh.hdf5'.format(mesh_name),
                      save_fiber_vector=save_fiber_vector, V=V_q)

#save the mesh to a XDMF file as well for easy viewing
mesh_filename_hdf5 = ('output/leftventricular_mesh/{}_mesh.hdf5'.format(mesh_name)) 
comm = mpi_comm_world()
mesh = Mesh(comm)  # check mesh type                   
h5file = HDF5File(comm, mesh_filename_hdf5, 'r') 
h5file.read(mesh, 'mesh', True) #read hdf5 file and store the mesh as a variable
with XDMFFile(comm, 'output/leftventricular_mesh/{}_mesh.xdmf'.format(mesh_name)) as f:
    f.write(mesh)

