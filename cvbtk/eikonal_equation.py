from dolfin import (Expression, Function, DirichletBC, Parameters, Constant, Identity, 
                    outer, inner, as_vector, dot, grad, TestFunction, NonlinearVariationalProblem,
                    NonlinearVariationalSolver, TrialFunction, derivative, solve, XDMFFile, File,
                    mpi_comm_world, HDF5File, FunctionSpace, VectorElement, Mesh, MeshFunction, VectorFunctionSpace)
from dolfin import *
import numpy as np
import math 

from ufl import Measure
# from cvbtk.dirichlet import get_dirichlet_bc
# from utils import vector_space_to_scalar_space
# from geometries import LeftVentricleGeometry


class EikonalProblem(object):

    def __init__(self, mesh, fibers, surface_tags):
        # fsn = model.geometry.fiber_vectors()
        # u = model.u

        V = FunctionSpace(mesh , 'Lagrange',1)
        # V = vector_space_to_scalar_space(u.ufl_function_space())
        self.V = V
        mesh = V.ufl_domain().ufl_cargo()
        self.mesh = mesh

        # self.t_dep = Function(V)
        # self.t_dep.assign(Expression('x[0]/1000000',element=V.ufl_element()))

        # self.dx = Measure('dx', subdomain_data=self.mesh.tags())
        # self.ds = Measure('ds', subdomain_data=self.mesh.tags())

        # self.dx = Measure('dx', domain = mesh, subdomain_data=surface_tags)
        # self.ds = Measure('ds', domain = mesh, subdomain_data=surface_tags)



        td0BC = Constant(0.0)
        # self.dirichlet = DirichletBC(V, td0BC, surface_tags)
        # Define the Dirichlet boundary conditions, if given.
        # bc_par = parameters['boundary_conditions']
        # dir_par = (bc_par['dirichlet']
        #            if bc_par is not None else None)
        # dir_par = None

        # # dir_par_dict = {'component': 0, 'g': 10.0, 'sub_domain': 0}
        # dir_par_dict = dir_par.to_dict()
        # if dir_par is not None:
        #     self.dirichlet = get_dirichlet_bc(
        #         V, dir_par_dict,
        #         mesh.surface_tags)
        # else:
        #     self.dirichlet = None
        # def boundary(x, on_boundary):
        #     a = Constant(43)
        #     sig = (1/(2*a))*( sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+a)*(x[2]+a))+ sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-a)*(x[2]-a)))
        #     tau = (1/(2*a))*( sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+a)*(x[2]+a))- sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-a)*(x[2]-a)))
        #     ksi = float(math.acosh(sig))
        #     phi = math.atan(x[1]/x[0])
        #     theta = float(math.acos(tau))
        #     x0 = 11.1
        #     y0 = -11.1
        #     z0 = -14
        #     x1 = 0.
        #     y1 = 16.3
        #     z1 = 0
        #     x2 = -16.3
        #     y2 = 0.
        #     z2 = 0
        #     x3 = -28.4
        #     y3 = 0
        #     z3 = -24
        #     x4 = 21.5
        #     y4 = 0
        #     z4 = 25
        #     r0 = sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)+(x[2]-z0)*(x[2]-z0))
        #     r1 = sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)+(x[2]-z1)*(x[2]-z1))
        #     r2 = sqrt((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2)+(x[2]-z2)*(x[2]-z2))
        #     r3 = sqrt((x[0]-x3)*(x[0]-x3)+(x[1]-y3)*(x[1]-y3)+(x[2]-z3)*(x[2]-z3))
        #     r4 = sqrt((x[0]-x4)*(x[0]-x4)+(x[1]-y4)*(x[1]-y4)+(x[2]-z4)*(x[2]-z4))
        #     ksi_endo = 0.375053614389
        #     on_endo = abs(ksi - ksi_endo ) <= 1.e-3
        #     ksi_epi = 0.685208
        #     on_epi = abs(ksi - ksi_epi ) <=1.e-3
        #     r_act = 4
        #     within_r0 = r0<= r_act
        #     within_r1 = r1<= r_act
        #     within_r2 = r2<= r_act
        #     within_r3 = r3<= r_act
        #     within_r4 = r4<= r_act
        #     return ( on_endo and ( within_r0 or within_r1 or within_r2 )) or ( on_epi and within_r3 ) #or within_r4

        sub_domains = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
        sub_domains.set_all(0)

        bound_points = boundary()
        bound_points.mark(sub_domains,1)

        file = File("subdomains.pvd")
        file << sub_domains
        self.dirichlet = DirichletBC(V, td0BC , sub_domains, 1, method ='pointwise')
        # self.dirichlet = DirichletBC(V, td0BC , boundary, method ='pointwise')

        self.parameters = self.default_parameters()
        prm = self.parameters
        
        self.I = Identity(3)
        # self.fibers = fsn
        self.fibers = fibers

        ef0 = as_vector([self.fibers[0],
                        self.fibers[1],
                        self.fibers[2]])
        self.ef0 = ef0/sqrt(inner(ef0, ef0))
        self.ef0ef0 = outer(self.ef0, self.ef0)
        self.Mi = Constant(prm['sig_il']) * self.ef0ef0 + Constant(prm['sig_it']) * (self.I-self.ef0ef0)
        self.Me = Constant(prm['sig_el']) * self.ef0ef0 + Constant(prm['sig_et']) * (self.I-self.ef0ef0) 

        self.M = self.Mi + self.Me

    # def define_linear(self):
    #     # q = TestFunction(self.V)
    #     prm = self.parameters

        w = TestFunction(self.V)
        td = Function(self.V)

        p = grad(td)
        beta_i = inner(self.Mi*p,p)/ inner(self.M*p,p)
        beta_e = inner(self.Me*p,p)/ inner(self.M*p,p)
        q = ( beta_i**2*self.Me+ beta_e**2*self.Mi)*p

        td00 = TrialFunction(self.V)
        sig = Constant(prm['sig_il']) * self.ef0ef0 + Constant(prm['sig_it']) *(self.I- self.ef0ef0)
        a_in = inner( grad(w), sig* grad( td00 ))*(1/Constant(prm['cm']))*dx
        a = inner( grad(w), q)*(1/Constant(prm['cm']))*dx+Constant(prm['rho'])* sqrt( inner(grad(td),q))*w*dx
        L = Constant(prm['f'])*w*dx + Constant(prm['g'])*w*ds
        F = a-L
        # Compute solution
        J = derivative(F, td)
        pde = NonlinearVariationalProblem(F, td , self.dirichlet , J)
        solver = NonlinearVariationalSolver(pde)

        # initialize with poisson solution
        # log(INFO , "Solving poisson to initialize")
        td0 = Function(self.V)
        solve( a_in ==L,td0 , self.dirichlet , solver_parameters ={"linear_solver": "mumps"})

        td0.rename('td0', 'td0')
        file = File('init_LV.pvd')
        file << td0



        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        td.assign(td0)
        ofile = XDMFFile(mpi_comm_world(), "Report_LV.xdmf")
        it = 0
        cm = prm['cm']
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

    @staticmethod
    def default_parameters():
        prm = Parameters('eikonal')

        prm.add('cm', (5*10**-4))
        prm.add('rho', (1.41421 * np.sqrt(1e3)))
        prm.add('sig_il', (2e-4))
        prm.add('sig_it', (4.16e-5))
        prm.add('sig_el', (2.5e-4))
        prm.add('sig_et', (1.25e-4))
        prm.add('f', (1))
        prm.add('g', (0))

        return prm

class boundary(SubDomain):
    """
    Helper method to mark the basal, inner and outer
    surface boundaries.
    """
    def __init__(self):
        self.parameters = self.default_parameters()
        self.point_prm = self.parameters['activation_points']
        self.tol = 1.e-3
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        prm = self.parameters
        point_prm = self.point_prm

        # C = Constant(prm['C'])
        # sig = (1/(2*C))*(sqrt(x[0]**2 + x[1]**2 + (x[2] + C)**2)
        #         + sqrt(x[0]**2 + x[1]**2 + (x[2] - C)**2))

                # tau = 0.5/C*(np.sqrt(x[0]**2 + x[1]**2 + (x[2] + C)**2)
        #         - np.sqrt(x[0]**2 + x[1]**2 + (x[2] - C)**2))

        # ksi = float(math.acosh(sig))
        # phi = atan(x[1]/x[0])
        # theta = float(acos(tau))
        xr, yr, zr = self.ellipsToCartesian(prm['epi_ksi'])
        print(xr,yr,zr)

        r0 = sqrt((x[0]-point_prm['x0'])**2 + (x[1]-point_prm['y0'])**2 + (x[2]-point_prm['z0'])**2)
        r1 = sqrt((x[0]-point_prm['x1'])**2 + (x[1]-point_prm['y1'])**2 + (x[2]-point_prm['z1'])**2)
        r2 = sqrt((x[0]-point_prm['x2'])**2 + (x[1]-point_prm['y2'])**2 + (x[2]-point_prm['z2'])**2)
        r3 = sqrt((x[0]-point_prm['x3'])**2 + (x[1]-point_prm['y3'])**2 + (x[2]-point_prm['z3'])**2)
        # r3 = sqrt((x[0]-xr)**2 + (x[1]-yr)**2 + (x[2]-zr)**2)

        # on_endo = abs(ksi - prm['endo_ksi']) <= self.tol
        # on_epi = abs(ksi - prm['epi_ksi']) <= self.tol

        r_act = point_prm['r_act']

        within_r0 = r0 <= r_act
        within_r1 = r1 <= r_act
        within_r2 = r2 <= r_act
        within_r3 = r3 <= r_act

        return on_boundary and ( within_r0 or within_r1 or within_r2 ) or (within_r3 )

        # return ( on_endo and ( within_r0 or within_r1 or within_r2 )) or ( on_epi and within_r3 )

    def ellipsToCartesian(self, ksi,theta=3/5*math.pi,phi = 1/2*math.pi):
        focus = self.parameters['C']
        e = ksi

        # b = focus * sqrt(1-e**2)/e
        a = focus/e
        b = sqrt((1-ksi**2) * a**2)
        eps = math.asinh(b/focus)

        x = focus * sinh(eps) * sin(theta) * cos(phi)
        y = focus * sinh(eps) * sin(theta) * sin(phi)
        z = focus * cosh(eps) * cos(theta)
        return x, y, z

    @staticmethod
    def default_parameters():
        prm = Parameters('left_ventricle')
        prm.add('C', 43)
        prm.add('endo_ksi', 0.375053614389)
        prm.add('epi_ksi', 0.685208)

        point_prm = Parameters('activation_points')

        point_prm.add("x0", 11.1)
        point_prm.add("y0", -11.1)
        point_prm.add("z0", -14)
        point_prm.add("x1", 0.)
        point_prm.add("y1", 16.3)
        point_prm.add("z1", 0)
        point_prm.add("x2", -16.3)
        point_prm.add("y2", 0.)
        point_prm.add("z2", 0)
        point_prm.add("x3", -28.4)
        point_prm.add("y3", 0)
        point_prm.add("z3", -24)
        point_prm.add('r_act', 4)

        prm.add(point_prm)

        return prm

if __name__ == '__main__':
    meshfile = "/home/maaike/Pleunie/sepran_tetra_82422_all_P1.h5"
    with HDF5File(mpi_comm_world(), meshfile, 'r') as f:
        mesh = Mesh(mpi_comm_world())
        f.read(mesh, 'mesh', True)  
        surface_tags = MeshFunction('size_t', mesh )
        VV = VectorFunctionSpace(mesh , 'Lagrange', 1, dim=3)
        fibers = Function(VV)
        f.read(fibers , 'fibers')
        f.read(surface_tags, 'boundaries') 
    # h5file = HDF5File( mpi_comm_world() , 'sepran_tetra_82422_all_P1.h5', 'r')
    # mesh = Mesh(mpi_comm_world())
    # h5file.read(mesh , 'mesh', True )
    # surface_tags = MeshFunction('size_t', mesh )
    # h5file.read( surface_tags , 'boundaries')
    # V = FunctionSpace(mesh , 'Lagrange',1)
    # log(INFO , "Function Space dimension {}".format(V.dim ()))

    # # Fibers
    # VV = VectorFunctionSpace(mesh , 'Lagrange', 1, dim=3)
    # fibers = Function(VV)
    # h5file.read(fibers , 'fibers')
    print(surface_tags[0])
    
    EikonalProblem(mesh, fibers, surface_tags)


