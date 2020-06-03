from dolfin import *
import numpy as np
import h5py

mesh2 = UnitCubeMesh(6, 6, 6)

f = h5py.File('td.hdf5','r')
f.visit(print)

openfile = HDF5File(mpi_comm_world(), 'td.hdf5', 'r')
attr = openfile.attributes('td')
element = attr['signature']
print(element)
family = element.split(',')[0].split('(')[-1][1:-1]
print(family)
cell = element.split(',')[1].strip()
print(cell)
degree = int((element.split(',')[2]).split(')')[0])
print(degree)
# quad_scheme = element.split(',')[3].split('=')[1].split(')')[0][1:-1]
element_V = VectorElement(family=family,
                            cell=cell,
                            degree=degree)
V = FunctionSpace(mesh2, element_V)
td = Function(V)



V2 = FunctionSpace(mesh2, "Lagrange", 1)
# VV = VectorFunctionSpace(mesh2, "Lagrange", 1, dim=3)

# parameters['allow_extrapolation'] = True
# u = Function(V2)
parameters['allow_extrapolation'] = False
u = project(td, V2)
file = File("projected.pvd")
file << u

# v2d = vertex_to_dof_map(V2)
values = u.vector().array()
print(len(values))
np.savetxt("td_values.csv", values, fmt='%f')

act = Function(V2)
max_td = max(values)
print('td max: {}'.format(max_td))
steps = 10
step_size = np.int(np.ceil(max_td/steps))
dt = step_size/2
for i in range (0,np.int(np.ceil(max_td))+1,step_size):
    idx = np.where((values > i -step_size) & (values <= i))[0]
    print('i: {}, idx act: {}'.format(i,idx))
    act.vector()[idx] = i

file = File("activation.pvd")
file << act
