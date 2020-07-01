# First, the :py:mod:`dolfin` module is imported: ::

from dolfin import *
import numpy as np
import csv
import os
from math import hypot
# Create mesh and define function space
LL = 1
# mesh = UnitCubeMesh(10, 10, 10)
for element in range(1,11):
    elem = element
    print(elem)
    dir_out = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/eikonal_box/boxmesh_boundary_effect'.format(elem)
    # mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(10., 4., 2.), 10, 10, 10)
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(5.0, 7.0, 1.35), (5 * elem), (7 * elem), (2 * elem))
    V = FunctionSpace(mesh, "Lagrange", 2)
    VV = VectorFunctionSpace(mesh, "Lagrange", 2, dim=3)

    print('cell size:',mesh.hmin())

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x, on_boundary):
        # center_x = 2.5
        # center_y = 3.5
        # height_z = 1.35-0.45
        # r = 1.
        # d = r**2 - (center_x -x[0])**2  + (center_y - x[1])**2
        # oncirclehaut = hypot(x[0]-center_x, x[1]-center_y)
        # return oncirclehaut > r/2.0 and on_boundary
        # return d > 0 and (x[2] > height_z)
        # return (x[0] > 2.15 and x[0] < 2.85) and (x[1] > 3. and x[1] < 4.) and (x[2] > height_z)
        tol = 1E-14
        return on_boundary and (near(x[1], 0, tol))

    # Define boundary condition
    td0 = Constant(0.0)
    bc = DirichletBC(V, td0, boundary)

    # Define parameters
    cm = Constant(5*10**-3) #1.e+3
    rho = Constant(1.41421)
    sig_il = Constant(2e-3)
    sig_it = Constant(4.16e-4)
    sig_el = Constant(2.5e-3)
    sig_et = Constant(1.25e-3)

    parameters["form_compiler"]["quadrature_degree"] = 2
    parameters["form_compiler"]["representation"] = "uflacs"

    # Define variational problem
    w = TestFunction(V)
    td = Function(V)
    f = Constant(1)
    g = Constant(0)
    fiber = Function(VV)
    # fiber_exp = Expression(("cos(0.5*pi*x[2]-0.25*pi)","sin(0.5*pi*x[2]-0.25*pi)","0."), element=VV.ufl_element())
    # fiber_exp = Expression(("-45*sin(4*pi*((x[2]/L)-0.5)*((x[2]/L)-0.5)*((x[2]/L)-0.5))","-45*sin(4*pi*((x[2]/L)-0.5)*((x[2]/L)-0.5)*((x[2]/L)-0.5))","0."), element=VV.ufl_element(), L=LL)
    # fiber_exp = Expression(("x[1]", "x[1]","0."), element=VV.ufl_element())
    fiber_exp = Expression(("0.", "x[1]","0."), element=VV.ufl_element())

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
    file = File(os.path.join(dir_out,"init_3D.pvd"))
    file << td0

    # solve the nonlinear problem
    log(INFO, "Solving eikonal equation")
    td.assign(td0)
    solver.solve()

    # Save solution in VTK format
    td.rename("td","td")
    file = File(os.path.join(dir_out,"td_3D_{}.pvd".format(elem)))
    file << td


# with HDF5File(mpi_comm_world(), 'td.hdf5', 'w') as f:
#     f.write(td, 'td')
#     f.write(mesh, 'mesh')
# # with HDF5File(mpi_comm_self(), meshfile, 'a') as f:


# mesh2 = UnitCubeMesh(6, 6, 6)
# V2 = FunctionSpace(mesh2, "Lagrange", 1)
# # VV = VectorFunctionSpace(mesh2, "Lagrange", 1, dim=3)

# # parameters['allow_extrapolation'] = True
# # u = Function(V2)
# parameters['allow_extrapolation'] = False
# u = project(td, V2)
# file = File("projected.pvd")
# file << u

# # v2d = vertex_to_dof_map(V2)
# values = u.vector().array()
# print(len(values))
# np.savetxt("td_values.csv", values, fmt='%f')

# act = Function(V2)
# max_td = max(values)
# print('td max: {}'.format(max_td))
# steps = 10
# step_size = np.int(np.ceil(max_td/steps))
# dt = step_size/2
# for i in range (0,np.int(np.ceil(max_td))+1,step_size):
#     idx = np.where((values > i -step_size) & (values <= i))[0]
#     print('i: {}, idx act: {}'.format(i,idx))
#     act.vector()[idx] = i

# file = File("activation.pvd")
# file << act
