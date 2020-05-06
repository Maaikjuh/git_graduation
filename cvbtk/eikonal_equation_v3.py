from dolfin import *
import math as math
import numpy as np

# Create mesh and define function space
h5file = HDF5File( mpi_comm_world() , "/home/maaike/model/cvbtk/data/mesh_leftventricle_30.hdf5", 'r')
mesh = Mesh(mpi_comm_world())
h5file.read(mesh , 'mesh', True )
# surface_tags = MeshFunction('size_t', mesh )
# h5file.read( surface_tags , 'boundaries')
V = FunctionSpace(mesh , 'Lagrange',1)
log(INFO , "Function Space dimension {}".format(V.dim ()))

# Fibers
VV = VectorFunctionSpace(mesh , 'Lagrange', 1, dim=3)
fibers = Function(VV)
h5file.read(fibers , 'fiber_vector')

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
    
    TODO   get **kwargs working 
           make dictionairy with infarct parameters in main to be passed 
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
    print(xi)
    phi0 = 1/2*math.pi
    phi1 = 1/4*math.pi
    phi2 = 0.
    theta0 = 1/2*math.pi

    xi_max = 0.67
    xi_min = 0.371

    tol = 1/20 * math.pi

    on_xi_max = xi < xi_max + tol and xi > xi_max - tol
    on_xi_min = xi < xi_min + tol and xi > xi_min - tol

    p0 = phi < phi0 + tol and phi > phi0 - tol and theta < theta0 + tol and theta > theta0 - tol 
    p1 = phi < phi1 + tol and phi > phi1 - tol and theta < theta0 + tol and theta > theta0 - tol 
    p2 = phi < phi2 + tol and phi > phi2 - tol and theta < theta0 + tol and theta > theta0 - tol 

    return (on_xi_min and (p0 or p1)) or (on_xi_max and p2) and on_boundary








# def boundary(x, on_boundary ):
#     a = Constant(4.3)
#     sig = (1/(2*a))*( sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+a)*(x[2]+a))+ sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-a)*(x[2]-a)))
#     tau = (1/(2*a))*( sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+a)*(x[2]+a))- sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-a)*(x[2]-a)))
#     ksi = float(math.acosh(sig))
#     phi = atan(x[1]/x[0])
#     theta = float(acos(tau))
#     x0 = 11.1/10
#     y0 = -11.1/10
#     z0 = -14/10
#     x1 = 0.
#     y1 = 16.3/10
#     z1 = 0
#     x2 = -16.3/10
#     y2 = 0.
#     z2 = 0
#     x3 = -28.4/10
#     y3 = 0
#     z3 = -24/10
#     x4 = 21.5/10
#     y4 = 0
#     z4 = 25/10
#     r0 = sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)+(x[2]-z0)*(x[2]-z0))
#     r1 = sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)+(x[2]-z1)*(x[2]-z1))
#     r2 = sqrt((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2)+(x[2]-z2)*(x[2]-z2))
#     r3 = sqrt((x[0]-x3)*(x[0]-x3)+(x[1]-y3)*(x[1]-y3)+(x[2]-z3)*(x[2]-z3))
#     r4 = sqrt((x[0]-x4)*(x[0]-x4)+(x[1]-y4)*(x[1]-y4)+(x[2]-z4)*(x[2]-z4))
#     ksi_endo = 0.934819
#     on_endo = abs(ksi - ksi_endo ) <= 1.e-3
#     ksi_epi = 0.807075
#     on_epi = abs(ksi - ksi_epi ) <=1.e-3
#     r_act = 4/10
#     within_r0 = r0<= r_act
#     within_r1 = r1<= r_act
#     within_r2 = r2<= r_act
#     within_r3 = r3<= r_act
#     within_r4 = r4<= r_act
#     return ( on_endo and ( within_r0 or within_r1 or within_r2 )) or ( on_epi and within_r3 ) #or within_r4

# phi = compute_coordinate_expression(degree, V,'phi',focus)
# theta = compute_coordinate_expression(degree, V,'theta',focus)
# xi = compute_coordinate_expression(degree, V,'xi',focus)

td0BC = Constant(0.0)
bc = DirichletBC(V, td0BC , boundary, method ='pointwise')

# Define parameters
cm = Constant(5*10**-4)
rho = Constant(1.41421 * sqrt(1e3))
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
ff = as_vector([ fibers[0], fibers[1], fibers[2]])
ef = ff/sqrt(inner(ff ,ff))
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
file = File("Report_init_LV.pvd")
file << td0

# solve the nonlinear problem
solver.parameters["newton_solver"]["linear_solver"] = "mumps"
log(INFO , "Solving eikonal equation")
td.assign(td0)
ofile = XDMFFile(mpi_comm_world(), "Report_LV.xdmf")
it = 0
dcm = ((6.7*10**-4)-(5*10**-4))/20
while float(cm) <= (6.7*10**-4):
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