from dolfin import *
import math as math
import numpy as np
import os
import csv

import cvbtk
from cvbtk.mechanics import compute_coordinate_expression
from cvbtk import LeftVentricleGeometry, read_dict_from_csv, save_to_disk, scalar_space_to_vector_space, save_dict_to_csv
from cvbtk import GeoFunc

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore')

#define parameters below

meshres = 20 # choose 20, 25, 30, 35, 40, 45 or 50
segments = 20

project_td = False
project_meshres = 20
project_segments = 20

# factor which specifies how much larger the conductivity
# in the purkinje fibers area ((sub)endocardial) is
# set to 1 if the conductivity should not be larger in the purkinje area
sig_fac_purk = 1.
radius = 0.5

sig_il = 1.5e-3 *0.3
sig_el = 1.5e-3*0.3
sig_it = 1.5e-3*0.3
sig_et = 1.5e-3*0.3

theta0 = 1/2*math.pi
theta1 = 1/2*math.pi
theta2 = 1/2*math.pi
theta3 = 1/2*math.pi
theta4 = 1/2*math.pi

phi0 = 0.
phi1 = 0.
phi2 = 0.
phi3 = 0.
phi4 = 0.

# directory where the outputs are stored
# now = datetime.datetime.now()
dirout = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/Eikonal_meshes/seg_{}_mesh_{}_bue_sig_higher'.format(segments,meshres)
dirout_project = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/Eikonal_meshes/seg_{}_mesh_{}_bue_default'.format(project_segments,project_meshres)
print('saving output to: ' + dirout)

# mesh that is selected for the calculation
filepath = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(segments,meshres)
filepath_project = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(project_segments,project_meshres)

class EikonalProblem(object):

    def __init__(self, dirout, filepath, **kwargs):
        self.parameters = self.default_parameters()
        self.parameters.update(kwargs)

        # compute epsilon from the eccentricities (given in leftventricular_meshing.py)
        e_outer = self.parameters['boundary']['outer_eccentricity']
        e_inner = self.parameters['boundary']['inner_eccentricity']
        eps_outer = self.cartesian_to_ellipsoidal(self.parameters['boundary']['focus'], eccentricity = e_outer)['eps']
        eps_inner = self.cartesian_to_ellipsoidal(self.parameters['boundary']['focus'], eccentricity = e_inner)['eps']

        self.parameters['boundary']['eps_outer'] = eps_outer
        self.parameters['boundary']['eps_inner'] = eps_inner

        self.dir_out = dirout

        #save parameters to csv   
        save_dict_to_csv(self.parameters, os.path.join(self.dir_out, 'inputs.csv'))
        save_dict_to_csv(self.parameters['boundary'], os.path.join(self.dir_out, 'inputs.csv'), write = 'a')
        # with open(os.path.join(self.dir_out, 'inputs.csv'), 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(self.parameters['boundary'])
        # save_dict_to_csv(self.parameters['boundary'], os.path.join(self.dir_out, 'inputs.csv'))

        #set parameters, quadrature degree should be the same to the loaded fiber vectors
        parameters["form_compiler"]["quadrature_degree"] = 4
        parameters["form_compiler"]["representation"] = "uflacs"

        geometry = LeftVentricleGeometry(meshfile=filepath)
        self.mesh = geometry.mesh()
        self.Q = FunctionSpace(self.mesh, 'Lagrange', 2)

        #extract fiber vectors
        print('extracting fiber vectors')
        self.ef = geometry.fiber_vectors()[0].to_function(None)

        #save loaded fiber vectors
        V1 = scalar_space_to_vector_space(self.Q)
        # ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"ef.xdmf"))
        # ofile.write(project(self.ef,V1))

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
        print('purkinje fibers')
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

            # ellipsoidal positions of the root points
            phi0 = prm['phi0']
            phi1 = prm['phi1']
            phi2 = prm['phi2']
            phi4 = prm['phi4']
            theta0 = prm['theta0']
            theta1 = prm['theta1']
            theta2 = prm['theta2']
            theta4 = prm['theta4']

            eps_outer = prm['eps_outer']

            # for de test, vergeet dit niet te uncommenten!!!
            # eps_outer= prm['eps_inner']


            eps_inner = prm['eps_inner']

            r = prm['radius']

            # get cartesian mid points of root points
            x0, y0, z0 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta0, phi0)
            x1, y1, z1 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta1, phi1)
            x2, y2, z2 = self.ellipsoidal_to_cartesian(focus, eps_outer, theta2, phi2)
            x3, y3, z3 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta2, phi2)
            x4, y4, z4 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta4, phi4)

            # check if a node is within the area of a root point (sphere)
            p0 = (x[0] - x0)**2 + (x[1] - y0)**2 + (x[2] - z0)**2 < r**2
            p1 = (x[0] - x1)**2 + (x[1] - y1)**2 + (x[2] - z1)**2 < r**2
            p2 = (x[0] - x2)**2 + (x[1] - y2)**2 + (x[2] - z2)**2 < r**2
            p3 = (x[0] - x3)**2 + (x[1] - y3)**2 + (x[2] - z3)**2 < r**2
            p4 = (x[0] - x4)**2 + (x[1] - y4)**2 + (x[2] - z4)**2 < r**2

            eps = self.cartesian_to_ellipsoidal(focus, x=x)['eps']

            # return (p0 or p1 or p3 or p4) and eps <= eps_inner + 0.001 or (p2 and eps >= eps_outer - 0.001)
            return ((p0 or p1 or p3 or p4) and on_boundary) or (p2 and on_boundary)
            # tol = 1.e-1
            # return (abs(x[1]) < tol and abs(x[2]-2.4) < tol and eps <= 0.3713 + tol) or (abs(x[0]) < tol and abs(x[2]-2.4) < tol and eps <= 0.3713 + tol)
            # return eps <= eps_inner + 0.01

        # define dirichlet boundary conditions -> td0BC for the stimulus locations
        print('Creating boundary conditions')
        bc = DirichletBC(self.Q, td0BC , boundary, method ='pointwise')
        # bc = DirichletBC(self.Q, td0BC , boundary)

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

        self.td = td

        self.td.rename('eikonal','eikonal')

        file = File(os.path.join(self.dir_out,"td_solution.pvd"))
        file << self.td

        # save mesh and solution for td as hdf5 to be used later on in the model
        with HDF5File(mpi_comm_world(), os.path.join(self.dir_out,'td.hdf5'), 'w') as f:
            f.write(td, 'td')
            f.write(self.mesh, 'mesh')
    
    def purkinje_fibers(self,sig,sigstr):
        focus = self.parameters['boundary']['focus']
        tol = self.parameters['boundary']['tol']
        eps_purk =  self.parameters['eps_purk']

        if sigstr == 'sig_il' or sigstr == 'sig_el':
            fac = 1.833
            # #test niet vergeten weg te halen!!
            # fac = 4.
        elif sigstr == 'sig_it' or sigstr == 'sig_et':
            fac = 2.667
        # fac = self.parameters['sig_fac_purk']

        sigval = self.parameters[sigstr]

        rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
        rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

        sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)
        
        eps = Expression("acosh({sigma})".format(sigma=sigmastr),degree=3,element=self.Q.ufl_element())

        purkinje_layer = "(eps > ({eps_purk}-{tol}) && eps < ({eps_purk}+{tol}))? ({sig}*{fac}) : {sig}".format(eps_purk=eps_purk, tol=tol, sig=sigval,fac=fac)
        purk_exp = Expression(purkinje_layer, element=self.Q.ufl_element(), eps=eps)

        sig.interpolate(purk_exp)

        # filename = os.path.join(self.dir_out, 'inputs.csv')
        sig_dict = {sigstr +'_purkinje': sigval*fac}
        save_dict_to_csv(sig_dict, os.path.join(self.dir_out, 'inputs.csv'), write = 'a')

        # with open(filename, 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([sigstr +'_purkinje', sigval*fac])

        return sig
    
    def cartesian_to_ellipsoidal(self, focus, x = [0,0,0], eccentricity = None):
        if eccentricity != None:
            x[0] = focus*np.sqrt(1-eccentricity**2)/eccentricity
            x[1] = 0
            x[2] = 0

        ra = sqrt(x[0]**2+x[1]**2+(x[2]+focus)**2)
        rb = sqrt(x[0]**2+x[1]**2+(x[2]-focus)**2)

        tau= (1./(2.*focus)*(ra-rb))
        sigma=(1./(2.*focus)*(ra+rb))

        expressions_dict = {"phi": math.atan2(x[1],x[0]),
                            "eps": math.acosh(sigma),
                            "theta": math.acos(tau)} 
        return expressions_dict
  
    def ellipsoidal_to_cartesian(self,focus, eps,theta,phi):
        x = focus * math.sinh(eps) * math.sin(theta) * math.cos(phi)
        y = focus * math.sinh(eps) * math.sin(theta) * math.sin(phi)
        z = focus * math.cosh(eps) * math.cos(theta) 

        return x, y, z

    @staticmethod
    def default_parameters():
        prm = Parameters('eikonal')
        prm.add('cm', (5*10**-3))
        prm.add('rho', (1.41421))
        prm.add('sig_fac_purk', 2.)
        prm.add('sig_il', (2e-3)*2)
        prm.add('sig_it', (4.16e-4)*2)
        prm.add('sig_el', (2.5e-3)*2)
        prm.add('sig_et', (1.25e-3)*2)
        prm.add('f', (1))
        prm.add('g', (0))
        prm.add('td0BC', 0.0)
        prm.add('eps_purk', 0.4)

        boundary_prm = Parameters('boundary')
        boundary_prm.add('focus', 4.3)
        boundary_prm.add('outer_eccentricity', 0.807075)
        boundary_prm.add('inner_eccentricity', 0.934819)
        boundary_prm.add('phi0', 0.)
        boundary_prm.add('phi1', -1/8*math.pi) 
        boundary_prm.add('phi2', 1/2*math.pi)    
        boundary_prm.add('phi4', -3/4*math.pi)   
        boundary_prm.add('theta0', 1/2*math.pi)
        boundary_prm.add('theta1', 9/16*math.pi)
        boundary_prm.add('theta2', 2/3*math.pi)
        boundary_prm.add('theta4', 1.2854)
        boundary_prm.add('eps_outer', float())
        boundary_prm.add('eps_inner', float())
        boundary_prm.add('radius', 0.5)   
        boundary_prm.add('tol', 1.e-1)
        boundary_prm.add('tol_theta', 1.e-1)

        prm.add(boundary_prm)
        return prm

    def project(self, filepath, dirout):
        print('projecting onto {}'.format(os.path.split(filepath)[1]))
        mesh = Mesh()

        openfile = HDF5File(mpi_comm_world(), filepath, 'r')
        openfile.read(mesh, 'mesh', False)
        V = FunctionSpace(mesh, "Lagrange", 2)

        self.td.set_allow_extrapolation(True)

        td_project = project(self.td, V)

        td_project.rename('eikonal','eikonal')

        file = File(os.path.join(dirout,"td_solution.pvd"))
        file << td_project

        # save mesh and solution for td as hdf5 to be used later on in the model
        with HDF5File(mpi_comm_world(), os.path.join(dirout,'td.hdf5'), 'w') as f:
            f.write(td_project, 'td')
            f.write(mesh, 'mesh')

if __name__ == '__main__':

    prm_boundary = {'radius': radius,
                    'phi0': phi0,
                    'phi1': phi1,
                    'phi2': phi2,
                    #'phi3': phi3,
                    'phi4': phi4,
                    'theta0': theta0,
                    'theta1': theta1,
                    'theta2': theta2,
                    #'theta3': theta3,
                    'theta4': theta4}

    inputs = {'sig_fac_purk': sig_fac_purk,
            'sig_il': sig_il,
            'sig_it': sig_it,
            'sig_el': sig_el,
            'sig_et': sig_et,
            'boundary': prm_boundary}

    # problem = EikonalProblem(dirout, filepath, **inputs)
    problem = EikonalProblem(dirout, filepath)
    problem.eikonal()

    if project_td == True:
        problem.project(filepath_project, dirout_project)


