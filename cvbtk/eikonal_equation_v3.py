from dolfin import *
import math as math
import numpy as np
import os
from cvbtk import LeftVentricleGeometry, read_dict_from_csv, save_to_disk, scalar_space_to_vector_space

# Create mesh and define function space
dir_out = 'eikonal'
filepath = 'mesh_leftventricle_30.hdf5' # "/home/maaike/model/cvbtk/data/mesh_leftventricle_30.hdf5"
h5file = HDF5File( mpi_comm_world() , filepath, 'r')
mesh = Mesh(mpi_comm_world())
h5file.read(mesh , 'mesh', True )
# surface_tags = MeshFunction('size_t', mesh )
# h5file.read( surface_tags , 'boundaries')
V = FunctionSpace(mesh , 'Lagrange',1)
log(INFO , "Function Space dimension {}".format(V.dim ()))



def compute_coordinate_expression(x,var,focus):
    """
    19-03 Maaike
    Expression for ellipsoidal coordinates (alternative definition)
    https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates

    Args:
        var : string of ellipsoidal coordinate to be calculated
        V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
        focus: Distance from the origin to the shared focus points.

    Returns:
        Expression for ellipsoidal coordinate
    """

    rastr = sqrt(x[0]**2+x[1]**2+(x[2]+focus)**2)
    rbstr = sqrt(x[0]**2+x[1]**2+(x[2]-focus)**2)

    taustr= (1./(2.*focus)*(rastr-rbstr))
    sigmastr=(1./(2.*focus)*(rastr+rbstr))

    expressions_dict = {"phi": math.atan2(x[1],x[0]),
                        "xi": math.acosh(sigmastr),
                        "theta": math.acos(taustr)} 
    return (expressions_dict[var])

def boundary(x, on_boundary):
    # V = FunctionSpace(mesh , 'Lagrange',1)
    focus = 4.3
    phi = compute_coordinate_expression(x,'phi',focus)
    theta = compute_coordinate_expression(x,'theta',focus)
    xi = compute_coordinate_expression(x,'xi',focus)
    phi0 = math.pi
    phi1 = -1/2*math.pi
    phi2 = 1/4*math.pi
    theta0 = 1/2*math.pi

    xi_max = 0.67
    xi_min = 0.371

    tol = 1.e-1
    tol_theta = 1.e-1

    on_xi_max = xi < xi_max + tol and xi > xi_max - tol
    on_xi_min = xi < xi_min + tol and xi > xi_min - tol

    p0 = phi < phi0 + tol and phi > phi0 - tol and theta < theta0 + tol_theta and theta > theta0 - tol_theta 
    p1 = phi < phi1 + tol and phi > phi1 - tol and theta < theta0 + tol_theta and theta > theta0 - tol_theta 
    p2 = phi < phi2 + tol and phi > phi2 - tol and theta < theta0 + tol_theta and theta > theta0 - tol_theta 

    return (on_xi_max and p2) or (on_xi_min and (p0 or p1))   

# Read inputs (we need the geometry inputs).
inputs = read_dict_from_csv('inputs.csv')

# Set the proper global FEniCS parameters.
parameters.update({'form_compiler': inputs['form_compiler']})

# Load geometry and first fiber field.

inputs['geometry']['load_fiber_field_from_meshfile'] = True
geometry = LeftVentricleGeometry(meshfile=filepath, **inputs['geometry'])
# ef = geometry.fiber_vectors()[0].to_function(None)
VV = VectorFunctionSpace(mesh , 'Lagrange', 1, dim=3)
fibers = Function(VV)
ef = geometry.fiber_vectors()[0].to_function(fibers)
# geometry._fiber_vectors = None  # This may be a redundant statement (but I did not check if it works without).
# geometry.load_fiber_field(filepath=filepath)
# Create scalar function space for the difference in angle between vectors.
Q = FunctionSpace(geometry.mesh(), 'Lagrange', 2)

# # Create vector function space.
# # Fibers
# VV = VectorFunctionSpace(mesh , 'Lagrange', 1, dim=3)
# fibers = Function(VV)
# ef = project(ef, VV)
V1 = scalar_space_to_vector_space(Q)
# ef = project(ef, V1)
save_to_disk(project(ef,V1), os.path.join(dir_out, 'ef.xdmf'))
# ofile = XDMFFile(mpi_comm_world(), "fibers.xdmf")
# ofile.write(ef)
fibers = ef

td0BC = Constant(0.0)
bc = DirichletBC(V, td0BC , boundary, method ='pointwise')

# Define parameters
cm = Constant(5*10**-2)
rho = Constant(1.4142)
sig_il = Constant(2e-3)
sig_it = Constant(4.16e-4)
sig_el = Constant(2.5e-3)
sig_et = Constant(1.25e-3)

# cm = Constant(5*10**-4)
# rho = Constant(1.41421 * sqrt(1e3))
# sig_il = Constant(2e-4)
# sig_it = Constant(4.16e-5)
# sig_el = Constant(2.5e-4)
# sig_et = Constant(1.25e-4)
parameters["form_compiler"]["quadrature_degree"] = 2
parameters["form_compiler"]["representation"] = "uflacs"

# Define variational problem
w = TestFunction(V)
td = Function(V)
f = Constant(1)
g = Constant(0)
# ff = as_vector([ fibers[0], fibers[1], fibers[2]])
# print(ff)
# ff = fibers
# ef = ff/sqrt(inner(ff ,ff))
I = Identity(3)
Mi = sig_il * outer(ef ,ef)+ sig_it *(I- outer(ef ,ef))
Me = sig_el * outer(ef ,ef)+ sig_et *(I- outer(ef ,ef))
M = Mi+Me
p = grad(td)
beta_i = inner(Mi*p,p)/ inner(M*p,p)
beta_e = inner(Me*p,p)/ inner(M*p,p)
q = ( beta_i**2*Me+ beta_e**2*Mi)*p

td00 = TrialFunction(V)
sig = sig_il * outer(ef ,ef)+ sig_it *(I- outer(ef ,ef))
a_in = inner( grad(w), sig* grad( td00 ))*(1/cm)*dx
a = inner( grad(w), q)*(1/cm)*dx+rho* sqrt( inner(grad(td),q))*w*dx
L = f*w*dx + g*w*ds
F = a-L

# Compute solution
J = derivative(F, td)
pde = NonlinearVariationalProblem(F, td , bc , J)
solver = NonlinearVariationalSolver(pde)

# initialize with poisson solution
log(INFO , "Solving poisson to initialize")
td0 = Function(V)
solve( a_in ==L,td0 , bc , solver_parameters ={"linear_solver": "mumps"})

# Save initilaization in VTK format
td0.rename("td0", "td0")
file = File(os.path.join(dir_out,"Report_init_LV.pvd"))
file << td0

# solve the nonlinear problem
solver.parameters["newton_solver"]["linear_solver"] = "mumps"
log(INFO , "Solving eikonal equation")
td.assign(td0)
ofile = XDMFFile(mpi_comm_world(), os.path.join(dir_out,"Report_LV.xdmf"))
it = 0
dcm = ((6.7*10**-2)-(5*10**-2))/20
while float(cm) <= (6.7*10**-2):
    print ("it={0}".format(it))
    print ("cm={0}".format(float(cm)))
    solver.solve()
    print(td.vector().norm('linf'))
    ofile.write(td , it , 0)
    td.assign(td)
    cm.assign(float(cm)+dcm)
    it = it+1

# ofile.write(td , it , 0)
ofile.close()
print('finished')
