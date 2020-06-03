# First, the :py:mod:`dolfin` module is imported: ::

from dolfin import *
import numpy as np
import csv

# Create mesh and define function space
# LL = 1
mesh = UnitCubeMesh(10, 10, 10)
V = FunctionSpace(mesh, "Lagrange", 1)
VV = VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return near(x[0], 0.5) and near(x[1], 0.5) and near(x[2], 1)

# Define boundary condition
td0 = Constant(0.0)
bc = DirichletBC(V, td0, boundary, method='pointwise')

# Define parameters
cm = Constant(8e-4) #1.e+3
rho = Constant(1.41421*sqrt(1e3))
sig_il = Constant(2e-4)
sig_it = Constant(4.16e-5)
sig_el = Constant(2.5e-4)
sig_et = Constant(1.25e-4)

parameters["form_compiler"]["quadrature_degree"] = 2
parameters["form_compiler"]["representation"] = "uflacs"

# Define variational problem
w = TestFunction(V)
td = Function(V)
f = Constant(1)
g = Constant(0)
fiber = Function(VV)
fiber_exp = Expression(("cos(0.5*pi*x[2]-0.25*pi)","sin(0.5*pi*x[2]-0.25*pi)","0."), element=VV.ufl_element())
#fiber_exp = Expression(("-45*sin(4*pi*((x[2]/L)-0.5)*((x[2]/L)-0.5)*((x[2]/L)-0.5))","-45*sin(4*pi*((x[2]/L)-0.5)*((x[2]/L)-0.5)*((x[2]/L)-0.5))","0."), element=VV.ufl_element(), L=LL)
fiber.assign(interpolate(fiber_exp, VV))
ff = as_vector([fiber[0], fiber[1], fiber[2]])
ef = ff/sqrt(inner(ff, ff))
I = Identity(3)
Mi = sig_il*outer(ef,ef)+sig_it*(I-outer(ef,ef))
Me = sig_el*outer(ef,ef)+sig_et*(I-outer(ef,ef))
M = Mi+Me
p = grad(td)
beta_i = inner(Mi*p,p)/inner(M*p,p)
beta_e = inner(Me*p,p)/inner(M*p,p)
q = (beta_i**2*Me+beta_e**2*Mi)*p

td00 = TrialFunction(V)
sig = sig_il*outer(ef,ef)+sig_it*(I-outer(ef,ef)) 
a_in = inner(grad(w), sig*grad(td00))*(1/cm)*dx
a = inner(grad(w), q)*(1/cm)*dx+rho*sqrt(inner(grad(td),q))*w*dx
L = f*w*dx + g*w*ds
F = a-L

# Compute solution
J = derivative(F, td)
pde = NonlinearVariationalProblem(F, td, bc, J)
solver = NonlinearVariationalSolver(pde)

# initialize with poisson solution
log(INFO, "Solving poisson to initialize")
td0 = Function(V)
solve(a_in==L,td0, bc)

# Save initilaization in VTK format
td0.rename("td0", "td0")
file = File("init_eikonal_3D_anisotropic_rotating.pvd")
file << td0

# solve the nonlinear problem
log(INFO, "Solving eikonal equation")
td.assign(td0)
solver.solve()

# Save solution in VTK format
td.rename("td","td")
file = File("eikonal_3D_anisotropic_rotating.pvd")
file << td


with HDF5File(mpi_comm_world(), 'td.hdf5', 'w') as f:
    f.write(td, 'td')



mesh2 = UnitCubeMesh(6, 6, 6)
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
