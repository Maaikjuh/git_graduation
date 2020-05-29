from dolfin import *
import math as math
import numpy as np
import os
import datetime
from cvbtk import LeftVentricleGeometry, read_dict_from_csv, save_to_disk, scalar_space_to_vector_space


class EikonalProblem(object):

    def __init__(self):
        self.parameters = self.default_parameters()

        now = datetime.datetime.now()
        self.dir_out = 'eikonal/{}_variable_test'.format(now.strftime("%d-%m_%H-%M"))

        filepath = 'mesh_leftventricle_30.hdf5' # "/home/maaike/model/cvbtk/data/mesh_leftventricle_30.hdf5"

        parameters["form_compiler"]["quadrature_degree"] = 4
        parameters["form_compiler"]["representation"] = "uflacs"

        # Read inputs (we need the geometry inputs).
        inputs = read_dict_from_csv('inputs.csv')

        inputs['geometry']['load_fiber_field_from_meshfile'] = True
        geometry = LeftVentricleGeometry(meshfile=filepath, **inputs['geometry'])
        self.ef = geometry.fiber_vectors()[0].to_function(None)
        self.Q = FunctionSpace(geometry.mesh(), 'Lagrange', 2)
        V1 = scalar_space_to_vector_space(self.Q)
        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"ef.xdmf"))
        ofile.write(project(self.ef,V1))
    
    def eikonal(self):
        prm = self.parameters
        cm = Constant(prm['cm'])
        rho = Constant(prm['rho'])
        sig_il = Constant(prm['sig_il'])
        sig_it = Constant(prm['sig_it'])
        sig_el = Constant(prm['sig_el'])
        sig_et = Constant(prm['sig_et'])
        f = Constant(prm['f'])
        g = Constant(prm['g'])
        td0BC = Constant(prm['td0BC'])

        def boundary(x, on_boundary):
            prm = self.parameters['boundary']
            focus = prm['focus']

            phi = self.compute_coordinate_expression(x,'phi',focus)
            theta = self.compute_coordinate_expression(x,'theta',focus)
            xi = self.compute_coordinate_expression(x,'xi',focus)

            phi0 = prm['phi0']
            phi1 = prm['phi1']
            phi2 = prm['phi2']
            theta0 = prm['theta0']

            xi_max = prm['xi_max']
            xi_min = prm['xi_min']

            tol = prm['tol']
            tol_theta = prm['tol_theta']

            on_xi_max = xi < xi_max + tol and xi > xi_max - tol
            on_xi_min = xi < xi_min + tol and xi > xi_min - tol

            p0 = phi < phi0 + tol and phi > phi0 - tol and theta < theta0 + tol_theta and theta > theta0 - tol_theta 
            p1 = phi < phi1 + tol and phi > phi1 - tol and theta < theta0 + tol_theta and theta > theta0 - tol_theta 
            p2 = phi < phi2 + tol and phi > phi2 - tol and theta < theta0 + tol_theta and theta > theta0 - tol_theta 

            return (on_xi_max and p2) or (on_xi_min and (p0 or p1)) 

        bc = DirichletBC(self.Q, td0BC , boundary, method ='pointwise')

        w = TestFunction(self.Q)
        td = Function(self.Q)
        td00 = TrialFunction(self.Q)

        ff = as_vector([ self.ef[0], self.ef[1], self.ef[2]])
        ef = ff/sqrt(inner(ff ,ff))
        I = Identity(3)
        Mi = sig_il * outer(ef ,ef)+ sig_it *(I- outer(ef ,ef))
        Me = sig_el * outer(ef ,ef)+ sig_et *(I- outer(ef ,ef))
        M = Mi+Me
        p = grad(td)
        beta_i = inner(Mi*p,p)/ inner(M*p,p)
        beta_e = inner(Me*p,p)/ inner(M*p,p)
        q = ( beta_i**2*Me+ beta_e**2*Mi)*p

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
        td0 = Function(self.Q)
        solve( a_in ==L,td0 , bc , solver_parameters ={"linear_solver": "mumps"})

        # Save initilaization in VTK format
        td0.rename("td0", "td0")
        file = File(os.path.join(self.dir_out,"Report_init_LV.pvd"))
        file << td0

        # solve the nonlinear problem
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        log(INFO , "Solving eikonal equation")
        td.assign(td0)
        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"Report_LV.xdmf"))
        it = 0
        solver.solve()
        print(td.vector().norm('linf'))
        ofile.write(td , it , 0)


    def compute_coordinate_expression(self,x,var,focus):
        """
        19-03 Maaike
        Expression for ellipsoidal coordinates (alternative definition)
        https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates

        Args:
            var : string of ellipsoidal coordinate to be calculated
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
        prm.add('td0BC', 0.0)

        boundary_prm = Parameters('boundary')
        boundary_prm.add('focus', 4.3)
        boundary_prm.add('phi0', math.pi)
        boundary_prm.add('phi1', -1/2*math.pi)
        boundary_prm.add('phi2', 1/4*math.pi)        
        boundary_prm.add('theta0', 1/2*math.pi)
        boundary_prm.add('xi_max', 0.67)
        boundary_prm.add('xi_min', 0.371)
        boundary_prm.add('tol', 1.e-1)
        boundary_prm.add('tol_theta', 1.e-1)
        

        prm.add(boundary_prm)
        return prm

if __name__ == '__main__':
    problem = EikonalProblem()
    problem.eikonal()


