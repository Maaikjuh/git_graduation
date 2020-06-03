# First, the :py:mod:`dolfin` module is imported: ::

from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(48, 48)
V = FunctionSpace(mesh, "Lagrange", 2)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return near(x[0], 0) and near(x[1], 0) #, DOLFIN_EPS)

# Define boundary condition
td_0 = Constant(0.0)
bc = DirichletBC(V, td_0, boundary, method='pointwise')

# Define parameters
cm = Constant(8e-4) #1.e+3
rho = Constant(50)
sig_il = Constant(2e-4)
sig_it = Constant(4.16e-5)
sig_el = Constant(2.5e-4)
sig_et = Constant(1.25e-4)

# Define variational problem
w = TestFunction(V)
td = Function(V)
f = Constant(1)
g = Constant(0)
ef = as_vector([1, -1])
I = Identity(2)
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

# Save initialization in VTK format
td0.rename("td0", "td0")
file = File("init_eikonal_2D_anisotropic_48.pvd")
file << td0

# solve the nonlinear problem
log(INFO, "Solving eikonal equation")
td.assign(td0)
solver.solve()

# Save solution in VTK format
td.rename("td","td")
file = File("eikonal_2D_anisotropic_48.pvd")
file << td