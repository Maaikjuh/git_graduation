"""
This script provides an example on how to create and export a left ventricular mesh.

"""

import time
import numpy as np
from dolfin.cpp.common import mpi_comm_world
from dolfin.cpp.mesh import * 
from dolfin import VectorElement, FunctionSpace
from dolfin.cpp.io import HDF5File, XDMFFile
from cvbtk import LeftVentricleGeometry, print_once, info_once, save_to_disk, quadrature_function_space
import csv

# We can inspect what parameters are passable to LeftVentricleGeometry:
print_once('Passable parameters to LeftVentricleGeometry:')
info_once(LeftVentricleGeometry.default_parameters(), True)

# ik heb zelf alleen nog maar 'mesh resolution' aangepast, ik heb nog niet gekeken wat de andere parameters precies deden
geometry_inputs = {
    'focus_height': 4.3,  
    'mesh_resolution': 30.,
    'inner_eccentricity': 0.934819,   
    'mesh_segments': 1,  
    'outer_eccentricity': 0.807075,  
    'truncation_height': 2.4}   

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

# Get the mesh coordinates. These are only printed to check the output and aren't stored (yet)
mesh_points=lv_mesh.coordinates()
print_once('coordinates: {} /n'.format(mesh_points))

# Save the mesh coordinates in a csv file
with open("mesh_coordinates.csv",'a',newline='') as outfile:
    writer = csv.writer(outfile)
    for row in mesh_points:
            writer.writerow([row])

# Get the order of degrees of freedom of functions from functions spaces over the mesh
# ik weet nog niet precies hoe dit precies werkt, of de 'CG' wel klopt en wat de dof mij precies zeggen 
V = FunctionSpace(lv_mesh, "CG",2)
n = V.dim()
d = lv_mesh.geometry().dim()

dof_coordinates = V.tabulate_dof_coordinates()
print_once('dof_coordinates: {} /n'.format(dof_coordinates))
dof_coordinates.resize((n, d))
print_once('dof_coordinates resized: {} /n'.format(dof_coordinates))

# Save the dof in a csv file
with open("dof_coordinates.csv",'a',newline='') as outfile:
    writer = csv.writer(outfile)
    for row in dof_coordinates:
            writer.writerow([row])


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

# Save the mesh to a XDMF file as well for easy viewing
mesh_filename_hdf5 = ('output/leftventricular_mesh/{}_mesh.hdf5'.format(mesh_name)) 
comm = mpi_comm_world()
mesh = Mesh(comm)  # check mesh type                   
h5file = HDF5File(comm, mesh_filename_hdf5, 'r') 
h5file.read(mesh, 'mesh', True) #read hdf5 file and store the mesh as a variable
with XDMFFile(comm, 'output/leftventricular_mesh/{}_mesh.xdmf'.format(mesh_name)) as f:
    f.write(mesh)

