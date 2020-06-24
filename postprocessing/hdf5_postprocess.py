from dolfin import *
from dolfin.cpp.common import mpi_comm_world
import numpy as np
import os

mesh = Mesh()
openfile = HDF5File(mpi_comm_world(),'/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref/mesh.hdf5', 'r')
openfile.read(mesh, 'mesh', False)
V = FunctionSpace(mesh, "Lagrange", 2)

active_stress = Function(V)

openfile = HDF5File(mpi_comm_world(), '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref/active_stress.hdf5', 'r')
openfile.read(active_stress, 'active_stress/vector_0')


