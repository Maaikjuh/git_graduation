from dolfin import *
from dolfin.cpp.common import mpi_comm_world
import numpy as np
import os
# from fenicstools import interpolate_nonmatching_mesh

# #eikonal mesh
mesh = Mesh()
openfile = HDF5File(mpi_comm_world(), '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/Eikonal_meshes/15-06_10-06_mesh_50_purk_fac_kot00/td.hdf5', 'r')
openfile.read(mesh, 'mesh', False)
V = FunctionSpace(mesh, "Lagrange", 2)
parameters['allow_extrapolation'] = True
td = Function(V)
openfile.read(td, 'td/vector_0')

mesh = Mesh()
openfile = HDF5File(mpi_comm_world(), '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/lv_maaike_seg20_res20_fibers_mesh.hdf5', 'r')
openfile.read(mesh, 'mesh', False)
V = FunctionSpace(mesh, "Lagrange", 2)
# V = Function(V)
parameters['allow_extrapolation'] = False
tact = project(-1*td,V)
print(tact)
tact.rename('eikonal', 'eikonal')
print(tact)
# # td.set_allow_extrapolation(True)
# V.interpolate(td)
# u2 = interpolate_nonmatching_mesh(td, V)
# file = File('eikonal.pvd')
# file << tact
tact_dummy = tact
file = XDMFFile('eikonal_td.xdmf')
for t in range(1,40,2):
    print(t)
    tact_dummy.vector()[:] += float(2)
    tact.assign(tact_dummy)
    
    file.write(tact, float(t))
# file.close()