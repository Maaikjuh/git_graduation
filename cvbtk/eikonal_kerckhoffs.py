from dolfin import *
import math as math
import numpy as np
import os
import csv

import cvbtk
from cvbtk.mechanics import compute_coordinate_expression
from cvbtk import LeftVentricleGeometry, read_dict_from_csv, save_to_disk, scalar_space_to_vector_space, save_dict_to_csv
from cvbtk import GeoFunc

#define parameters below
meshres = 30 # choose 20, 25, 30, 35, 40, 45, 50, 55, 60, 70 or 80
segments = 20

project_td = False
project_meshres = 20
project_segments = 20

# directory where the outputs are stored
file_name = 'kerckhoffs'

# for meshres in [20, 30, 40, 50, 60, 70, 80]:
    # now = datetime.datetime.now()
dirout = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/Eikonal_meshes/seg_{}_mesh_{}_{}'.format(segments,meshres, file_name)
dirout_project = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/Eikonal_meshes/seg_{}_mesh_{}_{}'.format(project_segments,project_meshres, file_name)
print('saving output to: ' + dirout)

# mesh that is selected for the calculation
filepath = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(segments,meshres)
filepath_project = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(project_segments,project_meshres)

class EikonalProblem(object):

    def __init__(self, dirout, filepath, **kwargs):
        self.parameters = self.default_parameters()
        self.parameters.update(kwargs)

        # compute epsilon from the eccentricities (given in leftventricular_meshing.py)
        # e_outer = self.parameters['boundary']['outer_eccentricity']
        # e_inner = self.parameters['boundary']['inner_eccentricity']
        # eps_outer = self.cartesian_to_ellipsoidal(self.parameters['boundary']['focus'], eccentricity = e_outer)['eps']
        # eps_inner = self.cartesian_to_ellipsoidal(self.parameters['boundary']['focus'], eccentricity = e_inner)['eps']

        # self.parameters['boundary']['eps_outer'] = eps_outer
        # self.parameters['boundary']['eps_inner'] = eps_inner

        self.dir_out = dirout

        #save parameters to csv   
        save_dict_to_csv(self.parameters, os.path.join(self.dir_out, 'inputs.csv'))
        save_dict_to_csv(self.parameters['geometry'], os.path.join(self.dir_out, 'inputs.csv'), write = 'a')
        save_dict_to_csv(self.parameters['root_points'], os.path.join(self.dir_out, 'inputs.csv'), write = 'a')

        #set parameters, quadrature degree should be the same to the loaded fiber vectors
        parameters["form_compiler"]["quadrature_degree"] = 4
        parameters["form_compiler"]["representation"] = "uflacs"

        geometry = LeftVentricleGeometry(meshfile=filepath)
        self.mesh = geometry.mesh()
        self.Q = FunctionSpace(self.mesh, 'Lagrange', 2)
        self.VV = TensorFunctionSpace(self.mesh, "Lagrange", 2)

        #extract fiber vectors
        print('extracting fiber vectors')
        self.ef = geometry.fiber_vectors()[0].to_function(None)
        self.es = geometry.fiber_vectors()[1].to_function(None)
        self.en = geometry.fiber_vectors()[2].to_function(None)

        ## Uncomment to save loaded fiber vectors to xdmf
        # self.V1 = scalar_space_to_vector_space(self.Q)
        # ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"ef.xdmf"))
        # ofile.write(project(self.ef,V1))

        self.c = Function(self.Q, name='c')
        self.c.assign(Constant(0.075))

        self.a = Function(self.Q, name='a')
        self.a.assign(Constant(0.4))
    
    def eikonal(self):
        prm = self.parameters
        ef = self.ef
        es = self.es
        en = self.en
        print(ef.vector())
        print(es.vector())
        print(en.vector())

        k= Constant(self.parameters['k_init']) #2.1e-2 #2.1
        c = self.purkinje_fibers(self.c, 'c')
        # c = self.parameters['cm'] #0.067 #(7.5) 
        a = self.purkinje_fibers(self.a, 'a')
        # a = self.parameters['am'] #(0.38)

        # Neumann boundary condition: g (0) on boundary, f in volume
        f = Constant(prm['f'])
        g = Constant(prm['g'])

        # value of activation time of the initial stimulated region(s)
        td0BC = Constant(prm['td0BC'])

        w = TestFunction(self.Q)
        td = Function(self.Q, name = 'td')

        # poisson equation
        td00 = TrialFunction(self.Q)
        # unknown initial guess
        td0 = Function(self.Q, name = 'td0')

        def boundary(x, on_boundary):
            prm = self.parameters['root_points']
            geom = self.parameters['geometry']
            focus = geom['focus']

            # ellipsoidal positions of the root points
            phi0 = prm['phi0']
            phi1 = prm['phi1']
            phi2 = prm['phi2']
            phi3 = prm['phi3']
            phi4 = prm['phi4']
            theta0 = prm['theta0']
            theta1 = prm['theta1']
            theta2 = prm['theta2']
            theta3 = prm['theta3']
            theta4 = prm['theta4']

            eps_outer = geom['eps_outer']
            eps_inner = geom['eps_inner']

            r = prm['radius']

            # get cartesian mid points of root points
            x0, y0, z0 = self.ellipsoidal_to_cartesian(focus, eps_outer, theta0, phi0)
            x1, y1, z1 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta1, phi1)
            x2, y2, z2 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta2, phi2)
            x3, y3, z3 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta3, phi3)
            # x4, y4, z4 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta4, phi4)

            # check if a node is within the area of a root point (sphere)
            p0 = (x[0] - x0)**2 + (x[1] - y0)**2 + (x[2] - z0)**2 < r**2
            # p0 = (x[1] - y0)**2 + (x[2] - z0)**2 < r**2
            p1 = (x[0] - x1)**2 + (x[1] - y1)**2 + (x[2] - z1)**2 < r**2
            p2 = (x[0] - x2)**2 + (x[1] - y2)**2 + (x[2] - z2)**2 < r**2
            p3 = (x[0] - x3)**2 + (x[1] - y3)**2 + (x[2] - z3)**2 < r**2
            # p4 = (x[0] - x4)**2 + (x[1] - y4)**2 + (x[2] - z4)**2 < r**2

            # eps = self.cartesian_to_ellipsoidal(focus, x=x)['eps']

            
            # return (p0 or p1 or p3 or p4) and eps <= eps_inner + 0.001 or (p2 and eps >= eps_outer - 0.001)
            return ((p0 or p1 or p2 or p3) and on_boundary) 
            # tol = 1.e-1
            # return p0 and (x[0] > 0)#and on_boundary
            # return (abs(x[1]) < tol and abs(x[2]-2.4) < tol and eps <= 0.3713 + tol) or (abs(x[0]) < tol and abs(x[2]-2.4) < tol and eps <= 0.3713 + tol)
            # return eps <= eps_inner + 0.01

        # define dirichlet boundary conditions -> td0BC for the stimulus locations
        print('Creating boundary conditions')
        # bc = DirichletBC(self.Q, td0BC , boundary, method ='pointwise')
        bc = DirichletBC(self.Q, td0BC , boundary)

        # initialize with poisson solution
        log(INFO , "Solving poisson to initialize") 

        # construct the conductivity tensor M from Mi and Me
        M = outer(ef ,ef)+ a * (outer(es, es) + outer(en, en))

        # unit normal to the wavefront, pointing away from depolarized tissue
        p = grad(td)

        # initial guess: Poisson equation
        a_in = inner(grad(w), (M * grad(td00))) * k * dx
        L = f*w*dx + g*w*ds

        # solver_linear = "mumps"
        solver_linear = 'mumps'      
        solve(a_in == L, td0, bc, solver_parameters = {"linear_solver": solver_linear})

        # Save initilaization in VTK format
        file = File(os.path.join(self.dir_out,"Init_td.pvd"))
        file << td0
        # Eikonal equation
        a = inner(grad(w), (M * p)) * k * dx + c * sqrt(inner(M * p, p))*w*dx

        F = a-L

        # Jacobian of F: dF/du
        J = derivative(F, td)

        # nonlinear variational problem: Find u (td) in V such that:
        # F(u; w) = 0 for all w in V^
        # solver relies on Jacobian (using Newton's method)
        pde = NonlinearVariationalProblem(F, td , bc , J)

        # create solver
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["linear_solver"] = solver_linear
        solver.parameters["newton_solver"]["maximum_iterations"] = 10

        # assign initial guess (td0) to td
        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"td_solution.xdmf"))

        td.assign(td0)
        k_end = self.parameters['k']
        k_init = self.parameters['k_init']
        dk = (float(k_end) - float(k_init))/20
        it = 0
        # iteratively solve Eikonal equation
        while float(k) >=  (float(k_end) - 0.0001):
            solver.solve()
            print('iteration ', it)
            print('td max = ', round(max(td.vector()),2))
            print('k = ', float(k))
            ofile.write(td, float(it))

            it += 1
            k.assign(float(k_init) + it * dk)
            td.assign(td)
            

        self.td = td

        ofile.close()

        # file = File(os.path.join(self.dir_out,"td_solution.pvd"))
        # file << self.td

        # save mesh and solution for td as hdf5 to be used later on in the model
        with HDF5File(mpi_comm_world(), os.path.join(self.dir_out,'td.hdf5'), 'w') as f:
            f.write(td, 'td')
            f.write(self.mesh, 'mesh')

        focus = prm['geometry']['focus']
        # get eps and phi values of the nodes in the mesh
        rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
        rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

        sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)
        eps = Expression("acosh({sigma})".format(sigma=sigmastr),degree=3,element=self.Q.ufl_element())

        td_inner_cpp = "(eps <= {eps_endo})? td : 0".format(eps_endo = prm['geometry']['eps_inner'])
        td_inner_exp = Expression(td_inner_cpp,degree=3,element=self.Q.ufl_element(), eps=eps, td=td)
        td_endo = Function(self.Q)
        td_endo.interpolate(td_inner_exp)
        print('Endocard activated in {max_endo}ms, whole LV activated in {max_lv}ms'.format(
            max_endo = round(max(td_endo.vector()), 2), max_lv = round(max(td.vector()), 2)))
  
    
    def purkinje_fibers(self,val,str_val):
        focus = self.parameters['geometry']['focus']

        # get eps and phi values of the nodes in the mesh
        rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
        rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

        sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)
        
        eps = Expression("acosh({sigma})".format(sigma=sigmastr),degree=3,element=self.Q.ufl_element())
        phi = Expression("atan2(x[1],x[0])",degree=3,element=self.Q.ufl_element())

        phi_func = Function(self.Q)
        phi_func.interpolate(phi)

        # define phi values where LV borders to the RV
        max_phi = max(phi_func.vector()) - 1/6* np.pi
        min_phi = max(phi_func.vector()) - 5/6* np.pi
        print(max_phi, min_phi)

        # define border of RV epicardium
        border = 1/12 * np.pi

        # define epsilons of endocardium, epicardium and midwall
        eps_inner = self.parameters['geometry']['eps_inner']
        # eps_sub_endo = self.parameters['geometry']['eps_sub_endo']
        eps_mid = self.parameters['geometry']['eps_mid']
        eps_outer = self.parameters['geometry']['eps_outer']

        eps_func = Function(self.Q)
        eps_func.interpolate(eps) 
        eps_outer = max(eps_func.vector())
        eps_inner = min(eps_func.vector())

        # sigval = self.parameters[sigstr]/3
        # define scaling parameters for purkinje system
        if str_val == 'c':
            bulk = self.parameters['cm']
            purk = self.parameters['ce']
        elif str_val == 'a':
            bulk = self.parameters['am']
            purk = self.parameters['ae'] 

        # purkinje layer at the endocardium. Exponentially increases from midwall to endocardium
        purk_layer_exp ='({bulk}*(1 + ({purk} - {bulk}) * ((eps - {eps_mid})/({eps_endo} - {eps_mid}))))'.format(
            bulk = bulk, purk = purk, eps_mid = eps_mid, eps_endo = eps_inner)
        # C++ syntax: (a)? b : c
        #   if a == True:
        #       b
        #   else:
        #       c
        purkinje_layer = "(eps <= {eps_mid})? {purk_layer_exp} : {bulk}".format(eps_mid = eps_mid, bulk = bulk,purk_layer_exp=purk_layer_exp)
        

        # interpolate Expressions on Functions
        purk_exp = Expression(purkinje_layer, element=self.Q.ufl_element(), degree=3, eps=eps)
        val.interpolate(purk_exp)


        # write sigvals to xdmf for easy viewing. 
        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"{}.xdmf".format(str_val)))
        ofile.write(val)

        return val
    
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
        prm.add('k_init', 1e-2) #5e-2 #2.1e-3
        prm.add('k', 2.1e-3) #2.1e-3
        prm.add('cm', 0.067) #0.075
        prm.add('am', 1/2.5)
        prm.add('ce', 0.4)
        prm.add('ae', 1/1.5)
        # prm.add('sig_fac_purk', 2.)
        # prm.add('sig_il', (2e-3)) #(2e-3)*2
        # prm.add('sig_it', (4.16e-4)) #(4.16e-4)*2
        # prm.add('sig_el', (2.5e-3)) #(2.5e-3)*2
        # prm.add('sig_et', (1.25e-3)) #(1.25e-3)*2
        prm.add('f', (1))
        prm.add('g', (0))
        prm.add('td0BC', 0.0)

        geometry_prm = Parameters('geometry')
        geometry_prm.add('eps_purk', 0.4)
        geometry_prm.add('focus', 4.3)
        geometry_prm.add('outer_eccentricity', 0.807075)
        geometry_prm.add('inner_eccentricity', 0.934819)
        geometry_prm.add('eps_outer', 0.6784)
        geometry_prm.add('eps_mid', 0.5305)
        geometry_prm.add('eps_sub_endo', 0.41787)
        geometry_prm.add('eps_inner', 0.3713)

        prm.add(geometry_prm)

        boundary_prm = Parameters('root_points')
        boundary_prm.add('phi0', 0.)
        boundary_prm.add('phi1', 0.) #-1/8*math.pi
        boundary_prm.add('phi2', -1/2*math.pi)    #1/2*math.pi
        boundary_prm.add('phi3', math.pi) 
        boundary_prm.add('phi4', -3/4*math.pi)   
        boundary_prm.add('theta0', 2/3*math.pi)#1/2*math.pi
        boundary_prm.add('theta1', 2/3*math.pi) #9/16*math.pi
        boundary_prm.add('theta2', 2/3*math.pi) #9/16*math.pi
        boundary_prm.add('theta3', 2/3*math.pi)
        boundary_prm.add('theta4', 1.2854)
        boundary_prm.add('radius', .5)   
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

    # problem = EikonalProblem(dirout, filepath, **inputs)
    problem = EikonalProblem(dirout, filepath)
    problem.eikonal()

    if project_td == True:
        problem.project(filepath_project, dirout_project)


