from dolfin import *
import math as math

# Create mesh and define function space
h5file = HDF5File( mpi_comm_world() , "/home/maaike/Pleunie/sepran_tetra_82422_all_P1.h5", 'r')
mesh = Mesh(mpi_comm_world())
h5file.read(mesh , 'mesh', True )
surface_tags = MeshFunction('size_t', mesh )
h5file.read( surface_tags , 'boundaries')
V = FunctionSpace(mesh , 'Lagrange',1)
log(INFO , "Function Space dimension {}".format(V.dim ()))

# Fibers
VV = VectorFunctionSpace(mesh , 'Lagrange', 1, dim=3)
fibers = Function(VV)
h5file.read(fibers , 'fibers')

# Define Dirichlet boundary
def boundary(x, on_boundary ):
    a = Constant(43)
    sig = (1/(2*a))*( sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+a)*(x[2]+a))+ sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-a)*(x[2]-a)))
    tau = (1/(2*a))*( sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+a)*(x[2]+a))- sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-a)*(x[2]-a)))
    ksi = float(math.acosh(sig))
    phi = atan(x[1]/x[0])
    theta = float(acos(tau))
    x0 = 11.1
    y0 = -11.1
    z0 = -14
    x1 = 0.
    y1 = 16.3
    z1 = 0
    x2 = -16.3
    y2 = 0.
    z2 = 0
    x3 = -28.4
    y3 = 0
    z3 = -24
    x4 = 21.5
    y4 = 0
    z4 = 25
    r0 = sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)+(x[2]-z0)*(x[2]-z0))
    r1 = sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)+(x[2]-z1)*(x[2]-z1))
    r2 = sqrt((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2)+(x[2]-z2)*(x[2]-z2))
    r3 = sqrt((x[0]-x3)*(x[0]-x3)+(x[1]-y3)*(x[1]-y3)+(x[2]-z3)*(x[2]-z3))
    r4 = sqrt((x[0]-x4)*(x[0]-x4)+(x[1]-y4)*(x[1]-y4)+(x[2]-z4)*(x[2]-z4))
    ksi_endo = 0.375053614389
    on_endo = abs(ksi - ksi_endo ) <= 1.e-3
    ksi_epi = 0.685208
    on_epi = abs(ksi - ksi_epi ) <=1.e-3
    r_act = 4
    within_r0 = r0<= r_act
    within_r1 = r1<= r_act
    within_r2 = r2<= r_act
    within_r3 = r3<= r_act
    within_r4 = r4<= r_act
    return ( on_endo and ( within_r0 or within_r1 or within_r2 )) or ( on_epi and within_r3 ) #or within_r4

# Define boundary conditions
td0BC = Constant(0.0)
bc = DirichletBC(V, td0BC , boundary , method ='pointwise')

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