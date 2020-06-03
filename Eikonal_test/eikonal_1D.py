# First, the :py:mod:`dolfin` module is imported: ::

from dolfin import *

# Create mesh and define function space
LL = 5
mesh = IntervalMesh(16,0,LL)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return near(x[0], 0)

# Define boundary condition
td_0 = Constant(0.0)
bc = DirichletBC(V, td_0, boundary)

# Define parameters
cm = Constant(8e-4)
rho = Constant(50)
sig = Constant(1e-4)

# Define variational problem
w = TestFunction(V)
td = TrialFunction(V)
f = Constant(1)
g = Constant(0) 
p = grad(td)

a = inner(grad(w), sig*p)*(1/cm)*dx+rho*sqrt(sig)*p[0]*w*dx
L = f*w*dx + g*w*ds

# Exact solution
aa = cm/sig
theta = rho*sqrt(sig)
td_exp = Expression("x[0]/theta-1/(a*theta*theta)*(exp(a*theta*(x[0]-L))-exp(-a*theta*L))", element=V.ufl_element(), theta=float(theta), a=float(aa), L=LL)
td_exact = interpolate(td_exp,V)

# Compute solution
td = Function(V)
solve(a == L, td, bc)

# compute difference and different norms of error
err = Function(V)
err.assign(td-td_exact)
err_norm_Linf = err.vector().norm('linf')
err_norm_L2 = sqrt(assemble(err*err*dx))
err_norm_H1 = sqrt(assemble(err*err*dx + inner(grad(err), grad(err))*dx))
print("Linf norm of Error is {}".format(err_norm_Linf))
print("L2 norm of Error is {}".format(err_norm_L2))
print("H1 norm of Error is {}".format(err_norm_H1))

# Save solution in VTK format
td.rename("td", "td")
td_exact.rename("td_exact","td_exact")
file = File("eikonal_1D_16e.pvd")
file << td
file << td_exact