from dolfin import *
import math as math
import numpy as np
import os
import datetime
import csv

import cvbtk
from cvbtk.mechanics import compute_coordinate_expression
from cvbtk import LeftVentricleGeometry, read_dict_from_csv, save_to_disk, scalar_space_to_vector_space, save_dict_to_csv
from cvbtk import GeoFunc

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore')

#define parameters below

meshres = 40 # choose 20, 25, 30, 35, 40, 45 or 50
segments = 20

# factor which specifies how much larger the conductivity
# in the purkinje fibers area ((sub)endocardial) is
# set to 1 if the conductivity should not be larger in the purkinje area
sig_fac_purk = 2.   

# directory where the outputs are stored
now = datetime.datetime.now()
dirout = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/Eikonal_meshes/seg_{}_mesh_{}_bue'.format(segments,meshres)

# mesh that is selected 
filepath = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(segments,meshres)

class EikonalProblem(object):

    def __init__(self, dirout, filepath, **kwargs):
        self.parameters = self.default_parameters()
        self.parameters.update(kwargs)
        
        self.dir_out = dirout

        #save parameters to csv   
        save_dict_to_csv(self.parameters, os.path.join(self.dir_out, 'inputs.csv'))
        save_dict_to_csv(self.parameters['boundary'], os.path.join(self.dir_out, 'inputs.csv'))

        #set parameters, quadrature degree should be the same to the loaded fiber vectors
        parameters["form_compiler"]["quadrature_degree"] = 4
        parameters["form_compiler"]["representation"] = "uflacs"

        geometry = LeftVentricleGeometry(meshfile=filepath)
        self.mesh = geometry.mesh()
        self.Q = FunctionSpace(self.mesh, 'Lagrange', 2)

        #extract fiber vectors
        self.ef = geometry.fiber_vectors()[0].to_function(None)

        #save loaded fiber vectors
        V1 = scalar_space_to_vector_space(self.Q)
        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"ef.xdmf"))
        ofile.write(project(self.ef,V1))

        #Initialize sigma vals, might be changed by purkinje definition
        self.sig_il = Function(self.Q)
        self.sig_it = Function(self.Q)
        self.sig_el = Function(self.Q)
        self.sig_et = Function(self.Q)

        self.sig_il.assign(Constant(self.parameters['sig_il']))
        self.sig_it.assign(Constant(self.parameters['sig_it']))
        self.sig_el.assign(Constant(self.parameters['sig_el']))
        self.sig_et.assign(Constant(self.parameters['sig_et']))
    
    def eikonal(self):
        prm = self.parameters
        ef = self.ef

        # membrane capacitance per unit tissue volume
        cm = Constant(prm['cm'])
        # ?
        rho = Constant(prm['rho'])

        # conductivity coefficients:
        #       intracellular(i), extracellular(e)
        #       parallel (l), perpendicular (t) to fiber directions
        sig_il = self.purkinje_fibers(self.sig_il, 'sig_il')
        sig_it = self.purkinje_fibers(self.sig_it, 'sig_it')
        sig_el = self.purkinje_fibers(self.sig_el, 'sig_el')
        sig_et = self.purkinje_fibers(self.sig_et, 'sig_et')

        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"sig_il.xdmf"))
        ofile.write(sig_il)

        # Neumann boundary condition: g (0) on boundary, f in volume
        f = Constant(prm['f'])
        g = Constant(prm['g'])

        # value of activation time of the initial stimulated region(s)
        td0BC = Constant(prm['td0BC'])

        w = TestFunction(self.Q)
        td = Function(self.Q)

        # poisson equation
        td00 = TrialFunction(self.Q)
        # unknown initial guess
        td0 = Function(self.Q)

        def boundary(x, on_boundary):
            prm = self.parameters['boundary']
            focus = prm['focus']

            phi = self.compute_coordinate_expression(x,'phi',focus)
            theta = self.compute_coordinate_expression(x,'theta',focus)
            xi = self.compute_coordinate_expression(x,'xi',focus)

            # position of the (three) locations of stimulus
            phi0 = prm['phi0']
            phi1 = prm['phi1']
            phi2 = prm['phi2']
            phi3 = prm['phi3']
            theta0 = prm['theta0']
            theta1 = prm['theta1']
            theta2 = prm['theta2']
            theta3 = prm['theta3']

            xi_max = prm['xi_max']
            xi_min = prm['xi_min']

            tol = prm['tol']
            tol_theta = prm['tol_theta']

            # check if a node is on the epicardium or on the endocardium
            on_xi_max = xi < xi_max + tol and xi > xi_max - tol
            on_xi_min = xi < xi_min + tol and xi > xi_min - tol

            # check if a node is within the specified values for phi for the stimulus locations
            p0 = phi < phi0 + tol and phi > phi0 - tol and theta < theta0 + tol_theta and theta > theta0 - tol_theta 
            p1 = phi < phi1 + tol and phi > phi1 - tol and theta < theta1 + tol_theta and theta > theta1 - tol_theta 
            p2 = phi < phi2 + tol and phi > phi2 - tol and theta < theta2 + tol_theta and theta > theta2 - tol_theta 
            p3 = phi < phi3 + tol and phi > phi3 - tol and theta < theta3 + tol_theta and theta > theta3 - tol_theta 

            # check if a node is within a specified location of stimulus
            return (on_xi_max and p2) or (on_xi_min and (p0 or p1 or p3)) 

        # define dirichlet boundary conditions -> td0BC for the stimulus locations
        bc = DirichletBC(self.Q, td0BC , boundary, method ='pointwise')

        # initial guess: Poisson equation
        I = Identity(3)
        sig = sig_il * outer(ef ,ef)+ sig_it *(I- outer(ef ,ef))
        a_in = inner( grad(w), sig* grad( td00 ))*(1/cm)*dx
        L = f*w*dx + g*w*ds

        # initialize with poisson solution
        log(INFO , "Solving poisson to initialize")       
        solve( a_in ==L,td0 , bc , solver_parameters ={"linear_solver": "mumps"})

        # Save initilaization in VTK format
        file = File(os.path.join(self.dir_out,"Init_td.pvd"))
        file << td0

        # construct the conductivity tensor M from Mi and Me
        Mi = sig_il * outer(ef ,ef)+ sig_it *(I- outer(ef ,ef))
        Me = sig_el * outer(ef ,ef)+ sig_et *(I- outer(ef ,ef))
        M = Mi+Me

        # unit normal to the wavefront, pointing away from depolarized tissue
        p = grad(td)

        #?
        beta_i = inner(Mi*p,p)/ inner(M*p,p)
        beta_e = inner(Me*p,p)/ inner(M*p,p)
        q = ( beta_i**2*Me+ beta_e**2*Mi)*p

        # Eikonal equation
        a = inner( grad(w), q)*(1/cm)*dx+rho* sqrt( inner(grad(td),q))*w*dx
        F = a-L

        # Jacobian of F: dF/du
        J = derivative(F, td)

        # nonlinear variational problem: Find u (td) in V such that:
        # F(u; w) = 0 for all w in V^
        # solver relies on Jacobian (using Newton's method)
        pde = NonlinearVariationalProblem(F, td , bc , J)

        # create solver
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"

        # assign initial guess (td0) to td
        td.assign(td0)
        # solve Eikonal equation 
        solver.solve()

        file = File(os.path.join(self.dir_out,"td_solution.pvd"))
        file << td

        # save mesh and solution for td as hdf5 to be used later on in the model
        with HDF5File(mpi_comm_world(), os.path.join(self.dir_out,'td.hdf5'), 'w') as f:
            f.write(td, 'td')
            f.write(self.mesh, 'mesh')
    
    def purkinje_fibers(self,sig,sigstr):
        focus = self.parameters['boundary']['focus']
        tol = self.parameters['boundary']['tol']
        xi_purk =  self.parameters['xi_purk']

        if sigstr == 'sig_il' or sigstr == 'sig_el':
            fac = 1.833
        elif sigstr == 'sig_it' or sigstr == 'sig_et':
            fac = 2.667
        # fac = self.parameters['sig_fac_purk']

        sigval = self.parameters[sigstr]

        xi = compute_coordinate_expression(3, self.Q.ufl_element(),'xi',focus)
        purkinje_layer = "(xi > ({xi_purk}-{tol}) && xi < ({xi_purk}+{tol}))? ({sig}*{fac}) : {sig}".format(xi_purk=xi_purk, tol=tol, sig=sigval,fac=fac)
        purk_exp = Expression(purkinje_layer, element=self.Q.ufl_element(), xi=xi)

        sig.interpolate(purk_exp)

        filename = os.path.join(self.dir_out, 'inputs.csv')
        with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([sigstr +'_purkinje', sigval*fac])

        # X = self.Q.tabulate_dof_coordinates().reshape(-1,3)
        # sigma_vals = 0.5 * (np.sqrt(X[:,0]** 2 + X[:,1] ** 2 + (X[:,2] + focus) ** 2) + np.sqrt(X[:,0] ** 2 + X[:,1] ** 2 + (X[:,2] - focus) ** 2)) / focus
        # sig_min = np.min(sigma_vals)
        # sig_max = np.max(sigma_vals)

        # v_wall_func = Function(self.Q)
        # v_wall = np.zeros(np.shape(self.sig_il.vector().array()))
        # for ii in range(len(self.sig_il.vector().array())):
        #     x, y, z = X[ii,:]

        #     sig_fun = GeoFunc.compute_sigma(x, y, z, focus)
        #     tau_fun = GeoFunc.compute_tau(x, y, z, focus)

        #     # Compute normalized radial coordinate v.
        #     v_wall_ii = GeoFunc.analytical_v(sig_fun, tau_fun, sig_min, sig_max)
        #     v_wall[ii] = v_wall_ii
        # print(sig_min,sig_max)
        # print(max(v_wall), min(v_wall))
        # v_wall_func.interpolate(v_wall)
        # sig_purk =  Expression("{sig_lt} * (1+(1.1/0.6-1)* (v_wall-0.3)/(1-0.5))".format(sig_lt = self.parameters[sigstr]),element=self.Q.ufl_element(),v_wall = v_wall_func)
        # purk_layer = Expression("(v_wall >= 0.5)? {sig_purk}: {sig_lt}".format(sig_purk=sig_purk, sig_lt=self.parameters[sigstr]),element=self.Q.ufl_element(),v_wall = v_wall_func)
        # sig.interpolate(purk_layer)

        return sig

    def compute_coordinate_expression_element(self,degree,element,var,focus):
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

        rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
        rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

        taustr= "(1./(2.*{f})*({ra}-{rb}))".format(f=focus,ra=rastr,rb=rbstr)
        sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)

        expressions_dict = {"phi": "atan2(x[1],x[0])",
                            "xi": "acosh({sigma})".format(sigma=sigmastr),
                            "theta": "acos({tau})".format(tau=taustr)} 
        return Expression(expressions_dict[var],degree=degree,element=element)


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
        prm.add('cm', (5*10**-3))
        prm.add('rho', (1.41421))
        prm.add('sig_il', (2e-3))
        prm.add('sig_it', (4.16e-4))
        prm.add('sig_el', (2.5e-3))
        prm.add('sig_et', (1.25e-3))
        prm.add('f', (1))
        prm.add('g', (0))
        prm.add('td0BC', 0.0)
        prm.add('xi_purk', 0.4)
        prm.add('sig_fac_purk', 2.)

        boundary_prm = Parameters('boundary')
        boundary_prm.add('focus', 4.3)
        boundary_prm.add('phi0', 0.)
        boundary_prm.add('phi1', -1/8*math.pi) #-1/2*math.pi
        boundary_prm.add('phi2', 1/2*math.pi)#1/4*math.pi      
        boundary_prm.add('phi3', -3/4*math.pi)   
        boundary_prm.add('theta0', 1/2*math.pi)
        boundary_prm.add('theta1', 9/16*math.pi)
        boundary_prm.add('theta2', 2/3*math.pi)
        boundary_prm.add('theta3', 1.2854)
        boundary_prm.add('xi_max', 0.67)
        boundary_prm.add('xi_min', 0.371)
        boundary_prm.add('tol', 1.e-1)
        boundary_prm.add('tol_theta', 1.e-1)
        

        prm.add(boundary_prm)
        return prm

if __name__ == '__main__':
    inputs = {'sig_fac_purk': sig_fac_purk}
    problem = EikonalProblem(dirout, filepath, **inputs)
    problem.eikonal()


