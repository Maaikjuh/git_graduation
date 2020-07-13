from dolfin import *
from dolfin.cpp.common import mpi_comm_world
import numpy as np
import os
import matplotlib.pyplot as plt

eikonal_dir = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/Eikonal_meshes/seg_20_mesh_20_bue/td.hdf5'
mesh_dir = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/lv_maaike_seg20_res20_fibers_mesh.hdf5'

mesh = Mesh()

openfile = HDF5File(mpi_comm_world(), mesh_dir, 'r')
openfile.read(mesh, 'mesh', False)
V = FunctionSpace(mesh, "Lagrange", 2)

openfile = HDF5File(mpi_comm_world(), eikonal_dir, 'r')
parameters['allow_extrapolation'] = True
td = Function(V)
openfile.read(td, 'td/vector_0')

fig = plt.figure()

roots = Expression('td < 10.', td = td, element = V.ufl_element())
print(roots)